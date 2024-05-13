# Copyright 2023 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script can be used to evaluate a model checkpoint.

# Example

P=languini/projects/gpt/logs/GPT_books16384_bsz512accum8_sl512_coslr0.0006to6e-06_h768_ff3072_nH12_dH64_nl12_clip0.0_decay36k_gpus2_defaultCompile_fp16
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/gpt/eval.py \
        --checkpoint_file "$P"/checkpoints/model.pt \
        --config_file "$P"/config.pickle \
        --eval_data_split test

"""

import os
import sys
import math
import torch
import pickle
import random
import argparse
import torch.multiprocessing as mp

from languini.train_lib import lm_trainer
from languini.train_lib import train_utils
from languini.dataset_lib import languini_books, multilingual
from languini.common_lib import debug_utils
from languini.common_lib import common_utils
from languini.common_lib import parallel_utils
from languini.common_lib import experiment_utils
from languini.common_lib.parallel_utils import mprint
from languini.common_lib.parallel_utils import LOCAL_RANK, WORLD_RANK, WORLD_SIZE

from model import Model

def run(config, language):
    c = config
    mprint(f"WORLD_SIZE: {WORLD_SIZE}")  # total number of devices
    mprint(f"WORLD_RANK: {WORLD_RANK}")  # unique id within all devices
    mprint(f"LOCAL_RANK: {LOCAL_RANK}")  # unique id within the devices of this node
    
    # load tokeniser
    sp = train_utils.load_tokeniser(name=c.dataset)

    # eval
    c.eval_batch_size = 1
    c.max_eval_steps = -1
    assert c.last_n > 0 and c.last_n <= c.seq_len and c.seq_len % c.last_n == 0, "Last_n has to be positive and a factor of sequence length."

    # Load the data split to evaluate on
    mprint("Setup data sources ... ")
    assert c.eval_batch_size % WORLD_SIZE == 0, "eval batch size has to be a multiple of the number of workers"
    # Compute the batch indices for this process.
    eval_batch_idxs = [i for i in range(c.eval_batch_size) if i % WORLD_SIZE == WORLD_RANK]
    END_OF_DOC_TOKEN = 2
    full_data_path = os.path.join(c.data_root, c.dataset)
    mprint(f"Loading split \"{c.eval_data_split}\" from {full_data_path}")
    
    # Compute the number of bytes in the data that we evaluate to correctly compute the normalised loss and ppl
    ds_args = dict(
        data_path=full_data_path,
        split=c.eval_data_split,
        repeat=False,
        global_batch_size=c.eval_batch_size,
        batch_idxs=eval_batch_idxs,
        micro_batches=1,
        sequence_length=c.seq_len,
        device=c.device,
        end_of_doc_token=END_OF_DOC_TOKEN,
        shift_n=c.last_n,
        sp=sp,
    )
    if c.num_cloned_languages > 0:
        # Cloned dataset
        assert language in ["L1", "L2"]
        assert not c.data_root_2
        ds = multilingual.ClonedLanguageDataset(
            num_languages=2, # HACK
            p_clone=0 if language == "L1" else 1,
            frac_clone=c.frac_clone,
            **ds_args,
        )
        # HACK: this (along with num_languages=2 above) is a nasty hack to always use L2 even if we have L3, L4, ...
        # TODO: solve cleaner
        ds.vocab_size = ds.original_vocab_size * (1 + c.num_cloned_languages)
    elif c.data_root_2:
        # Bilingual dataset
        assert language in ["L1", "L2"]
        full_data_path_2 = os.path.join(c.data_root_2, c.dataset_2)
        mprint(f"Loading \"{c.eval_data_split}\" from {full_data_path_2}")
        ds = multilingual.BilingualDataset(
            data_path_2=full_data_path_2,
            p_l2=0 if language == "L1" else 1,
            merge_vocab=c.merge_vocab,
            sp2=train_utils.load_tokeniser(name=c.dataset_2),
            **ds_args,
        )
    else:
        # Standard monolingual dataset
        assert not language
        ds = languini_books.LanguiniDatasetIterator(**ds_args)
    assert c.get("vocab_size") is None or c.vocab_size == ds.vocab_size
    c.vocab_size = ds.vocab_size

    # Build model and load it from checkpoint
    torch.manual_seed(c.seed)
    model = Model(config=c)
    if c.compile != "None":
        model = torch.compile(model, mode=c.compile)
    model = model.to(c.device)
    device_ids = [LOCAL_RANK] if c.device.type == "cuda" else None
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)  # we always use DDP so we can easily load models 
    model, curr_state = train_utils.load_checkpoint(model, c.checkpoint_file)
    mprint(f"Model checkpoint and state loaded from {c.checkpoint_file}")

    mprint("Measure test data size ...")
    eval_bytes, batch_count, token_count = lm_trainer.log_eval_stats(eval_data_source=ds,
                                                                        eval_steps=c.max_eval_steps,
                                                                        last_n=c.last_n,
                                                                        logger=None,
                                                                        device=c.device)
    mprint(f"number of bytes: {eval_bytes:,}")
    mprint(f"number of batches: {batch_count:,}")
    mprint(f"number of tokens: {token_count:,}")
    mprint(f"average bytes per token: {eval_bytes / token_count:.4f}")
    
    # Compute the loss of the model. Potentially by evaluating only the last few tokens. 
    mprint("Evaluate the model ...")
    mprint(f"sequence length: {c.seq_len}")
    mprint(f"evalulate last_n tokens per batch: {c.last_n}")

    eval_total_loss, eval_total_topk, eval_token_count, _ = lm_trainer.evaluation(config=c,
                                                                                    model=model,
                                                                                    state=curr_state,
                                                                                    data_source=ds,
                                                                                    max_steps=batch_count,
                                                                                    last_n=c.last_n,
                                                                                    print_progress=True)
    # loss and ppl over number of tokens
    eval_avg_loss = eval_total_loss / eval_token_count
    eval_ppl = math.exp(eval_avg_loss)
    # loss and ppl over number of bytes
    eval_norm_loss = eval_total_loss / eval_bytes
    eval_norm_ppl = math.exp(eval_norm_loss)
    # accuracy over tokens
    eval_topk_accs = {key: eval_total_topk[key] / eval_token_count for key in eval_total_topk.keys()}

    mprint(f"number of tokens: {eval_token_count:,}")
    warning_str = ""
    if token_count != eval_token_count:
        mprint(f"WARNING: token count to measure string length ({token_count:,}) DOES NOT EQUAL token count of the total loss evaluation ({eval_token_count:,})!")
        warning_str = "<- INVALID DUE TO MISMATCH IN TOKEN COUNT"
    mprint(f"total loss: {eval_total_loss}")
    mprint(f"avg loss: {eval_avg_loss:.6f}")
    mprint(f"ppl: {eval_ppl:.6f}")
    mprint(f"normalised loss: {eval_norm_loss:.6f} {warning_str}")
    mprint(f"normalised ppl: {eval_norm_ppl:.6f} {warning_str}")
    for key in eval_topk_accs.keys():
        mprint(f"top-{key} accuracy: {eval_topk_accs[key]:.6f}")
    
    mprint("Done!")

    return {
        "num_tokens": int(eval_token_count),
        "total_loss": float(eval_total_loss),
        "avg_loss": float(eval_avg_loss),
        "ppl": float(eval_ppl),
        "normalised_loss": float(eval_norm_loss),
        "normalised_ppl": float(eval_norm_ppl),
        **{f"top-{k}": float(v) for k, v in eval_topk_accs.items()},
    }


def main():
    """Load relevant args and evaluate on some data split."""
    
    # initialise distributed processes
    device = parallel_utils.init_distributed()
    mp.set_start_method("spawn")

    mprint("Languini Evaluation")

    # create parser and add args specific to eval
    parser = argparse.ArgumentParser(description='Runs evaluations.', usage=f"eval.py [<args>]")  
    parser.add_argument("--data_root", default=None, type=str, help="Path to the data in case it ought to differ from the path in the loaded config.")
    parser.add_argument("--checkpoint_file", default="", type=str, help=f"Model checkpoint to load.")
    parser.add_argument("--config_file", default="", type=str, help=f"Model config to load.")
    parser.add_argument("--wandb_run", default="", type=str, help=f"Wandb run to load model config and checkpoint from.")
    parser.add_argument("--eval_data_split", default="test", type=str, help=f"Name of the languini books split to do eval on.")
    parser.add_argument("--last_n", default=-1, type=int, help=f"Last n tokens to evaluate in the sequence.")
    parser.add_argument("--language", default="", type=str, help=f"Language to evaluate in. Required for multilingual runs.")
    args = parser.parse_args(sys.argv[1:])

    # download file from wandb if necessary
    if args.wandb_run:
        assert not args.checkpoint_file and not args.config_file, "Cannot load both from wandb and local filesystem."
        args.checkpoint_file, args.config_file = experiment_utils.load_wandb_checkpoint_and_config(args.wandb_run)

    # load config file
    with open(args.config_file, "rb") as f:
        config = pickle.load(f)
    mprint(f"original experiment name: {config.exp_name}")

    if args.data_root:
        config.data_root = args.data_root
    else:
        assert os.path.exists(config.data_root), f"The data root in the loaded config file does not exist ({config.data_root}). Set a custom data root using --data_root."
    config.checkpoint_file = args.checkpoint_file
    config.eval_data_split = args.eval_data_split
    config.last_n = args.last_n if args.last_n > 0 else config.seq_len
    config.device = device

    metrics = run(config, language=args.language)

    # name the results
    results_identifier = f"{args.eval_data_split}_{args.last_n}"
    if args.language:
        results_identifier += f"_{args.language}"

    if args.wandb_run:
        print("Uploading results to wandb ...")
        # update the run's summary metrics
        experiment_utils.log_wandb_summary_metrics(
            args.wandb_run,
            {f"{results_identifier}/{k}": v for k, v in metrics.items()}
        )

    print("Done.")


if __name__ == "__main__":
    main()
