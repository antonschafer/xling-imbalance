from collections import defaultdict
import os
import sys
import torch
import pickle
import argparse
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import networkx as nx

from languini.train_lib import train_utils
from languini.common_lib.debug_utils import check
from languini.common_lib import experiment_utils
from languini.common_lib.parallel_utils import mprint
from languini.dataset_lib.parallel_dataset import ParallelEn1En2Dataset, ParallelEnFrDataset
from languini.dataset_lib.multilingual import CombinedTokenizer

from model import Model


def compute_similarities(model, parallel_data_source, n_steps, matching_type):
    """
    Computes the cosine similarity between the hidden states and the gradients of a model on parallel data

    Args:
        model: the model to evaluate
        parallel_data_source: data source with parallel sequences
        num_steps: the number of steps to run
        matching_type: how to match the tokens of the two sequences that are compared with each other. Options are:
            - "one_by_one": compare the hidden states of the two sequences one by one
            - "max_matching": compute a maximum matching in the bipartite graph with similarity as edge weights
                TODO this is really slow and memory heavy right now
    
    Returns: Tuple
        hidden_state_cossims: array of cosine similaritites (n_layers, n_steps)
        gradient_cossims: a dictionary: parameter name -> array with similarities (n_steps,)
    """
    # prepare model
    model.train()
    model.zero_grad()

    # prepare datasets
    parallel_data_source.reset()

    # track similarities
    gradient_cossims = defaultdict(list)
    hidden_state_cossims = []
        
    for step in tqdm(range(n_steps)):
        grads_step = []
        hiddens_step = []

        batch_1, batch_2 = next(parallel_data_source)
        for batch_x, batch_y, is_padded in [batch_1, batch_2]:
            bsz, seqlen = batch_x.shape

            # compute grads
            with torch.cuda.amp.autocast(enabled = batch_x.device.type == "cuda"):
                # compute grads
                logits, _, all_hidden_states = model(batch_x, state=None, return_hidden=True)
                all_losses = F.cross_entropy(input=logits.view(-1, logits.shape[-1]), target=batch_y.flatten(), reduction='none')
                # ignore padding
                all_losses = torch.where(batch_y.flatten() == 0, 0, all_losses)
                loss = all_losses.mean()
                loss.backward()

                # retrieve grads
                curr_grads = {k: v.grad.clone().flatten() for k, v in model.named_parameters()}
                grads_step.append(curr_grads)
                model.zero_grad()

                # retrieve hidden states
                n_layers, _bsz, _seqlen, hdim = all_hidden_states.shape
                assert bsz == 1, "not implemented"
                all_hidden_states = all_hidden_states.squeeze(1)
                check(all_hidden_states, (n_layers, seqlen, hdim))
                num_nonpad = (batch_x != 0).sum()
                assert num_nonpad > 0
                assert batch_x[0, num_nonpad - 1] != 0, "padding in middle of sequence"
                hiddens_step.append(all_hidden_states[:, :num_nonpad, :].contiguous())

            
        # compute grad similarities
        for k in grads_step[0].keys():
            gradient_cossims[k].append(float(F.cosine_similarity(grads_step[0][k], grads_step[1][k], dim=0)))

        # compute hidden state similarities
        hs_1, hs_2 = hiddens_step # n_layers, seqlen_1/2, hdim
        if matching_type == "one_by_one":
            assert hs_1.shape == hs_2.shape, "sequences have to be of the same length for one_by_one matching"
            hidden_state_cossims.append(F.cosine_similarity(hs_1, hs_2, dim=2).mean(dim=1))
        elif matching_type == "max_matching":
            S1, S2 = hs_1.shape[1], hs_2.shape[1]
            cossim_alltoall = F.cosine_similarity(hs_1.unsqueeze(2), hs_2.unsqueeze(1), dim=3)
            check(cossim_alltoall, (n_layers, S1, S2))
            # match according to max average similarity over layers
            matching_mask = compute_max_matching(cossim_alltoall.mean(dim=0).cpu().numpy()).to(hs_1.device)
            # mask[i, j] true iff i-th token in sequence 1 matched with j-th token in sequence 2
            check(matching_mask, (S1, S2))
            assert matching_mask.sum() > 0
            # take mean over matched tokens in each layer
            cossim_layerwise = (cossim_alltoall * matching_mask.unsqueeze(0)).sum(dim=(1, 2)) / matching_mask.sum()
            hidden_state_cossims.append(cossim_layerwise)
        else:
            raise ValueError(f"Unknown matching type \"{matching_type}\".")


    return (
        torch.stack(hidden_state_cossims, dim=1).cpu().numpy(), # n_layers, n_steps 
        {k: np.array(v) for k, v in gradient_cossims.items()},
    )


def compute_max_matching(sim_AB):
    """
    Computes a maximum weight matching in a bipartite graph 

    Args:
        sim_AB: similarity scores, i.e. edge weights; shape (A, B)
    
    Returns:
        mask: binary mask of the matching; shape (A, B)
    """
    A, B = sim_AB.shape
    G = nx.Graph()
    for a in range(A):
        for b in range(B):
            ia, ib = a, b + A
            G.add_edge(ia, ib, weight= -sim_AB[a, b])
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(G, weight='weight')
    mask = torch.zeros(A, B, dtype=torch.bool)
    for ia, ib in matching.items():
        if ia > ib:
            continue
        a, b = ia, ib - A
        mask[a, b] = True
    return mask


def run(config, checkpoint_file, n_steps):
    c = config
    
    # Build model and load it from checkpoint
    torch.manual_seed(c.seed)
    model = Model(config=c)
    if c.compile != "None":
        model = torch.compile(model, mode=c.compile)
    model = model.to(c.device)
    model = torch.nn.DataParallel(model) # just for checkpoint keys to match
    model, _ = train_utils.load_checkpoint(model, checkpoint_file)
    mprint(f"Model checkpoint and state loaded from {checkpoint_file}")

    sp = train_utils.load_tokeniser(name=c.dataset)

    # prepare parallel data source
    if c.num_cloned_languages > 0:
        # Cloned dataset
        assert not c.data_root_2
        parallel_dataset = ParallelEn1En2Dataset(
            batch_size=1,
            device=c.device,
            original_vocab_size=sp.vocab_size()
        )
        matching_type = "one_by_one" # tokens in cloned languages are aligned
    elif c.data_root_2:
        # Bilingual dataset
        if not (c.dataset.startswith("books") and c.dataset_2.startswith("french_books")):
            raise NotImplementedError("Only EN-FR parallel data is supported at the moment.")
        sp2 = train_utils.load_tokeniser(name=c.dataset_2)
        parallel_dataset = ParallelEnFrDataset(
            batch_size=1,
            device=c.device,
            combined_tokeniser=CombinedTokenizer(sp, sp2, merge_vocab=c.merge_vocab),
        )
        matching_type = "max_matching" # tokens in different langauges are not aligned -> compute matching
    else:
        raise ValueError("No multilingual model")


    hidden_sims, grad_sims = compute_similarities(model, parallel_dataset, n_steps, matching_type)
    return hidden_sims, grad_sims


def main():
    mprint("Computing cosine similarities of hidden states and gradients when processing parallel sequences")

    # create parser and add args specific to eval
    parser = argparse.ArgumentParser(description='Runs evaluations.', usage=f"eval.py [<args>]")  
    parser.add_argument("--checkpoint_file", default="", type=str, help=f"Model checkpoint to load.")
    parser.add_argument("--config_file", default="", type=str, help=f"Model config to load.")
    parser.add_argument("--wandb_run", default="", type=str, help=f"Wandb run to load model config and checkpoint from.")
    parser.add_argument("--n_steps", default=500, type=int, help=f"Number of sequences to process.")
    parser.add_argument("--out_dir", default="", type=str, help=f"Local directory to save results to.")
    args = parser.parse_args(sys.argv[1:])

    # download file from wandb if necessary
    if args.wandb_run:
        assert not args.checkpoint_file and not args.config_file, "Cannot load both from wandb and local filesystem."
        args.checkpoint_file, args.config_file = experiment_utils.load_wandb_checkpoint_and_config(args.wandb_run)

    # load config file
    with open(args.config_file, "rb") as f:
        config = pickle.load(f)
    mprint(f"original experiment name: {config.exp_name}")

    hidden_sims, grad_sims = run(config, checkpoint_file=args.checkpoint_file, n_steps=args.n_steps)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        np.save(os.path.join(args.out_dir, "hidden_sims.npy"), hidden_sims)
        with open(os.path.join(args.out_dir, "grad_sims.pkl"), "wb") as f:
            pickle.dump(grad_sims, f)

    # display summary
    print("Hidden state mean cosine similarities")
    for i, layer_sims in enumerate(hidden_sims):
        print(f"\tLayer {i}: {layer_sims.mean():.4f} +/- {layer_sims.std():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()