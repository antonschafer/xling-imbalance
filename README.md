# Language Imbalance Can Boost Cross-lingual Generalisation

Code to the paper [Language Imbalance Can Boost Cross-lingual Generalisation](https://arxiv.org/abs/2404.07982). The code is based on the [Languini Kitchen](https://github.com/languini-kitchen/languini-kitchen), a codebase for training language models. For an overview of the changes made to support training on cloned languages and on French data, see [this diff](https://github.com/antonschafer/xling-imbalance/pull/1/files) or check out the [multilingual datasets](./languini/dataset_lib/multilingual.py).

## Reproducing Plots
To reproduce the plots from the paper without retraining models, you can load the relevant results via
```
wget https://y5d6.c15.e2-3.dev/public-bucket/results_xling.zip -O results_xling.zip
unzip results_xling.zip
```
then install languini in a new environment
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -e . --upgrade
```
and run the [analysis notebook](analysis.ipynb).

## Training and Evaluating Models

You can also train models on cloned languages or on French data as in the paper. Note that the numbers may not match exactly as the code cleanup slightly altered the randomness, yet the trends should be consistent. 

### Setup
Setup the environment as described above and load and tokenize the datasets:
- For the English data, follow the [languini instructions for obtaining the books3 dataset](#download-and-tokenise-the-books3-dataset).
- For the French data, run `./languini/dataset_lib/download_and_tokenise_french.sh`. Note that this will pip install packages.
(french data not required for cloned language experiments)
- For the parallel English-French dataset, follow the instructions in [languini/dataset_lib/parallel_dataset.py](languini/dataset_lib/parallel_dataset.py).
(parallel data only required for comparing hidden stats & gradients across languages)

### Training
Train models as described in Languini. Configure the language splits via the config arguments
- For cloned languages:
    - `num_cloned_languages`: number of cloned languages to introduce. E.g., to train on $\mathrm{EN}_1, \mathrm{EN}_2$, introduce $\mathrm{EN}_2$ by setting `num_cloned_languages = 1`.
    - `p_clone`: probability of sampling from the cloned language. E.g., when training with $\mathrm{EN}_1$ and $\mathrm{EN}_2$, `p_clone` = $\mathrm{EN}_2/\mathrm{EN}_1$. For >2 cloned languages, we use a cloned language which probability `p_clone` and sample uniformly which one to use.
    - `frac_clone`: fraction of the vocabulary to clone.
- For extra real languages (only French supported):
    - `data_root_2`: `"data/french-pd-books"` for french.
    - `dataset_2`: `"french_books_16384"` for data tokenized with the vocabulary of size 16384 used in the paper.
    - `p_l2`: probability of sampling from the second language.
    - `merge_vocab`: whether to merge the vocabularies of the two languages' tokenizers.
- For a schedule on `p_clone` / `p_l2` in the cloned / real language case, respectively, use `language_schedule`. Set e.g. `language_schedule="0.1_0.9_0.5_0.5"` for four evenly spaced stages with p = 10%, 90%, 50%, 50%.
    
For example, to train a model on $\mathrm{EN}_1$ and $\mathrm{EN}_2$ that coresponds to row 5 in Table 1, run
```
TRAIN_STEPS=18265
ACC_STEPS=8 # this works for a 4090 (24 GB)

torchrun --standalone languini/projects/gpt/main.py small \
    --train_batch_size 128 \
    --gradient_accumulation_steps $ACC_STEPS \
    --decay_steps $TRAIN_STEPS \
    --max_train_steps $TRAIN_STEPS \
    --num_cloned_languages 1  \
    --p_clone 0.1 \
    --frac_clone 1.0
```
For a model on $\mathrm{EN}$ and $\mathrm{FR}$ with a merged vocabulary and a language schedule that correponds to row 16 in Table 3, run

```
TRAIN_STEPS=18265
ACC_STEPS=8 # this works for a 4090 (24 GB)

torchrun --standalone languini/projects/gpt/main.py small \
    --train_batch_size 128 \
    --gradient_accumulation_steps $ACC_STEPS \
    --decay_steps $TRAIN_STEPS \
    --max_train_steps $TRAIN_STEPS \
    --data_root_2 "data/french-pd-books" \
    --dataset_2 "french_books_16384" \
    --p_l2 0.05 \
    --merge_vocab \
    --language_schedule "0.05_0.65_0.65_0.65"
```
### Evaluation
 Evaluate the model as described in Languini. You additionally have to specify the language to evaluate in as either `"L1"` (for $\mathrm{EN}_1$) or `"L2"` (for $\mathrm{EN}_2$ in the cloned setting or $\mathrm{FR}$ in the French setting) via the `--language` argument. E.g., to evaluate the run above on the French test data, run
```
RUN_PATH="path/of/your/wandb/run" # alternatively specify checkpoint_file and config_file

./venv/bin/torchrun --standalone languini/projects/gpt/eval.py \
    --wandb_run $RUN_PATH \
    --eval_data_split test \
    --last_n 128 \
    --language L2
```
if you specify a wandb run, this will automatically load the checkpoint from the run and finally upload the results to the run's summary.

### Representation Similarity
To compare a model's hidden states and gradients when fed parallel sequences in its two languages ($\mathrm{EN}_1, \mathrm{EN}_2$ or $\mathrm{EN}, \mathrm{FR}$, depending on how it was trained) run
```
python -m languini.projects.gpt.compare_representations --wandb_run "path/of/your/wandb/run" --out_dir "some/dir"

# or specify checkpoint_file and config_file instead of wandb_run
```

## Other Experiments
You can also reproduce the GLUE experiments via e.g.
```
./finetune_glue.sh path/of/your/wandb/run 
```

or the Word2Vec experiments via e.g.
```
python -m languini.projects.word2vec.main --p_clone 0.5
```



