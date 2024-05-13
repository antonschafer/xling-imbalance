# runs word2vec on 2 cloned languages. E.g. python -m languini.projects.word2vec.main --p_clone 0.5

import os
import itertools
import argparse

import pickle
import gensim
from tqdm import tqdm
import torch

from languini.dataset_lib import multilingual
from languini.train_lib import train_utils


def main():
    parser = argparse.ArgumentParser()
    # cloning args
    parser.add_argument('--p_clone', type=float, default=0.0)
    # word2vec args
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--window', type=int, default=5) # corresponds to default in gensim
    parser.add_argument('--sample', type=float, default=0.001) # corresponds to default in gensim
    parser.add_argument('--alpha', type=float, default=0.025) # corresponds to default in gensim
    parser.add_argument('--min_alpha', type=float, default=0.0001) # corresponds to default in gensim
    parser.add_argument('--negative', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    # data args -- do not change
    parser.add_argument('--train_steps', type=int, default=18265)
    parser.add_argument('--dataset', type=str, default="books_16384")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=512)
    # output args
    parser.add_argument('--output_dir', type=str, default=None)
    
    # setup config
    config = parser.parse_args()

    # setup output dir
    if config.output_dir is None:
        config.output_dir = f"word2vec_results/{config.p_clone}/"
    os.makedirs(config.output_dir, exist_ok=False)

    # load data
    END_OF_DOC_TOKEN = 2
    train_ds = multilingual.ClonedLanguageDataset(
        num_languages=2,
        p_clone=config.p_clone,
        frac_clone=1.0,
        sp=train_utils.load_tokeniser(config.dataset),
        data_path=os.path.join("data/books", config.dataset),
        split='train',
        repeat=True,
        global_batch_size=config.batch_size,
        batch_idxs=list(range(config.batch_size)),
        micro_batches=1,
        sequence_length=config.sequence_length,
        device="cpu",
        end_of_doc_token=END_OF_DOC_TOKEN,
    )
    final_vocab_size = train_ds.vocab_size # vocab size after cloning
    train_ds = itertools.islice(train_ds, config.train_steps)
    
    # convert to list of "sentences" for gensim
    # treat each sequence as a sentence and each token as a word
    sentences = []
    print("Processing dataset ...")
    for batch_x, batch_y, is_padded in tqdm(train_ds, total=config.train_steps):
        assert not is_padded
        new_sentences = batch_x.squeeze(0).tolist()
        sentences += new_sentences
    
    # train word2vec
    w2v_model = gensim.models.Word2Vec(
        min_count=1,
        window=config.window,
        vector_size=config.dim,
        sample=config.sample,
        alpha=config.alpha,
        min_alpha=config.min_alpha,
        negative=config.negative,
        workers=config.workers,
        seed=config.seed,
    )
    print("Building vocab ...")
    w2v_model.build_vocab(tqdm(sentences))
    print("Training word2vec model ...")
    w2v_model.train(tqdm(sentences), total_examples=len(sentences), epochs=1)
    
    print("Saving ...")
    # gather embeddings
    input_embeddings = torch.full((final_vocab_size, config.dim), fill_value=float("nan"))
    output_embeddings = torch.full((final_vocab_size, config.dim), fill_value=float("nan"))
    for subword_id in w2v_model.wv.index_to_key:
        input_embeddings[subword_id] = torch.tensor(w2v_model.wv[subword_id])
        output_embeddings[subword_id] = torch.tensor(w2v_model.syn1neg[w2v_model.wv.key_to_index[subword_id]])
    
    # save
    torch.save(input_embeddings, os.path.join(config.output_dir, "input_embeddings.pt"))
    torch.save(output_embeddings, os.path.join(config.output_dir, "output_embeddings.pt"))
    w2v_model.save(os.path.join(config.output_dir, "word2vec.model"))

    with open(os.path.join(config.output_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print("Done. Results saved to", config.output_dir)

if __name__ == "__main__":
    main()
