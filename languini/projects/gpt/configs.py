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

from munch import Munch  # Munch is a dictionary that supports attribute-style access


config_names = [
    'mini',
    'tiny',
    'small',
    'medium',
    'large',
    'XL',
]


def add_exp_name(config):
    """Constructs the name of the log folder used to easily identify the experiment. """
    c = config
    c.exp_name = ("GPT{}_{}_bsz{}{}_sl{}_coslr{}to{}_h{}_ff{}_nH{}_dH{}_nl{}_clip{}_decay{}k_workers{}_ncloned:{}_pclone:{}_fclone:{}_dr2:{}_ds2:{}_pl2:{}_merge:{}_ls:{}{}_fp16_seed{}{}"
                  .format("_flash" if c.use_flash else "",
                          c.dataset.replace("_", ""),
                          c.train_batch_size,
                          "" if c.gradient_accumulation_steps == 1 else f"_micro{c.gradient_accumulation_steps}",
                          c.seq_len,
                          c.max_lr,
                          c.min_lr,
                          c.h_dim,
                          c.mlp_dim,
                          c.n_heads,
                          c.head_dim,
                          c.n_layers,
                          c.grad_clip_norm,
                          c.decay_steps // 1_000,
                          c.n_workers,
                          c.num_cloned_languages,
                          c.p_clone,
                          c.frac_clone,
                          c.data_root_2.replace("_", ""),
                          c.dataset_2.replace("_", ""),
                          c.p_l2,
                          c.merge_vocab,
                          c.language_schedule,
                          "" if c.compile == "None" else f"_{c.compile}Compile",
                          c.seed,
                          f"_{c.comment}" if c.comment else "",
                          "_debug" if c.debug else "")                          
                 )


## Add experiment configs
def load_config(name=None):

    c = Munch(
        # data
        data_root = "data/books",
        relative_log_path = "logs",         # Relative path to the log folder within the project folder languini-kitchen/projects/gpt/logs/
        dataset = "books_16384",
        vocab_size = None,                  # to be set by the dataset
        debug = False,                      # simply adds a "_debug" suffix so logs are easily distinguishable

        # cloned languages
        num_cloned_languages = 0,           # how many cloned languages to add
        p_clone = 0.0,                        # proability of sampling from the cloned languages (uniform from L2, L3, L4, ... if num_cloned_languages > 1)
        frac_clone = 0.0,                     # fraction of the vocabulary that is cloned

        # multiple languages
        data_root_2 = "",                    # e.g. "data/french-pd-books"
        dataset_2 = "",                      # e.g. "french_books_16384",
        p_l2 = 0.0,                          # probability of sampling from the second language/dataset
        merge_vocab = False,                 # whether to merge the vocabularies of the two languages/tokenizers

        # scheduler of language ratio
        language_schedule = "",            # e.g. 0.1_0.9_0.5_0.5 for 4 stages of 10%, 90%, 50%, 50% p_l2/p_clone respectively

        # optimiser
        seed = 0,
        gradient_accumulation_steps = 1,    # number of batches before doing a gradient step
        train_batch_size = 16,              # make sure batch sizes are an integer multiple of the number of workers
        eval_batch_size = 16,
        test_batch_size = 16,
        seq_len = 512,
        max_eval_steps = 500,
        max_train_steps = 500_000,          # total number of training steps
        decay_steps = 500_000,              # number of steps over which we will decay the learning rate
        max_lr = 0.0006,                    # starting learning rate
        min_lr = 0.000006,                  # final learning rate
        grad_clip_norm = 0.0,               # gradient norm clipping
        tokens_per_second = 0,              # tokens per second throughput of this config on the hardware run; used for logging over gpuhours

        # perform certain tasks every N steps
        eval_every = 1_000,                 # perform a fast evaluation
        log_terminal_every = 100,           # print the current loss to terminal
        log_metrics_every = 100,            # log accuracy and loss metrics
        log_grads_every = 1_000,            # log gradients and step sizes
        log_activations_every = -1,         # log gradients and step sizes
        log_ckpt_every = 5_000,             # save model checkpoint to disk

        # logging
        comment = "",
        logger_type = 'all',  # can be 'tb', 'wandb' or 'all'
        wandb_project_name = 'gpt',
    )
    # default model
    if not name or name == 'default':
        name = 'mini'

    # model
    c.use_flash = False
    if name == 'mini':
        c.n_layers = 4
        c.h_dim = 512
        c.mlp_dim = 2048
        c.head_dim = 32
        c.n_heads = 8
    elif name == 'tiny':
        c.n_layers = 4
        c.h_dim = 768
        c.mlp_dim = 3072
        c.head_dim = 64
        c.n_heads = 12
    elif name == 'small':
        c.n_layers = 12
        c.h_dim = 768
        c.mlp_dim = 3072
        c.head_dim = 64
        c.n_heads = 12
    elif name == 'medium':
        c.n_layers = 24
        c.h_dim = 1024
        c.mlp_dim = 4096
        c.head_dim = 64
        c.n_heads = 16
    elif name == 'large':
        c.n_layers = 24
        c.h_dim = 1536
        c.mlp_dim = 6144
        c.head_dim = 96
        c.n_heads = 16
    elif name == 'XL':
        c.n_layers = 24
        c.h_dim = 2048
        c.mlp_dim = 8192
        c.head_dim = 128
        c.n_heads = 24
    else:
        raise ValueError(f"Config name {name} is an invalid name. ")

    return c
