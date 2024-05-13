# Contains code to preprocess and use the european parliament dataset with parallel lines in English and French.

# To be able to use this dataset:
# 1. Download English-French corpus from https://www.statmt.org/europarl/, unzip and place in data/fr-en
#   mkdir data/fr-en
#   cd data/fr-en
#   wget https://www.statmt.org/europarl/v7/fr-en.tgz
#   tar -xvf fr-en.tgz
# 2. Run this script to preprocess and tokenize the dataset


import os
import sentencepiece as spm
import torch


def load_and_preprocess(dataset_dir="data/fr-en", vocab_size=16384):
    output_file_en = os.path.join(dataset_dir, f"tokenised_en_{vocab_size}.pt")
    output_file_fr = os.path.join(dataset_dir, f"tokenised_fr_{vocab_size}.pt")
    if os.path.exists(output_file_en) or os.path.exists(output_file_fr):
        raise FileExistsError("Output files already exist")

    # load strings
    fr_file = os.path.join(dataset_dir, "europarl-v7.fr-en.fr")
    en_file = os.path.join(dataset_dir, "europarl-v7.fr-en.en")
    with open(fr_file, "r") as f:
        fr_lines = [x[:-1] for x in f.readlines()]
    with open(en_file, "r") as f:
        en_lines = [x[:-1] for x in f.readlines()]

    # load tokenisers
    vocab_en_path = os.path.join(os.getcwd(), 'languini/vocabs/spm_models', f"books_{vocab_size}.model")
    vocab_fr_path = os.path.join(os.getcwd(), 'languini/vocabs/spm_models', f"french_books_{vocab_size}.model")
    tokeniser_en = spm.SentencePieceProcessor()
    if not tokeniser_en.Load(vocab_en_path):
        raise Exception("Couldn't load english tokeniser.")
    tokeniser_fr = spm.SentencePieceProcessor()
    if not tokeniser_fr.Load(vocab_fr_path):
        raise Exception("Couldn't load french tokeniser.")

    # tokenise strings
    tokenised_en = tokeniser_en.encode(en_lines)
    tokenised_fr = tokeniser_fr.encode(fr_lines)
   
    # combine short lines into sequences
    seq_en, seq_fr = [[]], [[]]
    max_seqlen = 512
    min_seqlen = 128
    for en, fr in zip(tokenised_en, tokenised_fr):
        if max(len(en), len(fr)) > max_seqlen:
            continue
        if len(seq_en[-1]) + len(en) <= max_seqlen and len(seq_fr[-1]) + len(fr) <= max_seqlen:
            # add to current sequence
            seq_en[-1] += en
            seq_fr[-1] += fr
        else:
            if min(len(seq_en[-1]), len(seq_fr[-1])) < min_seqlen:
                # discard
                seq_en.pop()
                seq_fr.pop()
            # start new sequence
            seq_en.append(en)
            seq_fr.append(fr)

    print(f"Combined {len(tokenised_en)} lines into {len(seq_en)} sequences")
    assert len(seq_en) == len(seq_fr)

    # store as tensors (with 0 padding)
    tensor_en = torch.zeros((len(seq_en), max_seqlen), dtype=torch.long)
    tensor_fr = torch.zeros((len(seq_fr), max_seqlen), dtype=torch.long)
    for i, (en, fr) in enumerate(zip(seq_en, seq_fr)):
        tensor_en[i, :len(en)] = torch.tensor(en)
        tensor_fr[i, :len(fr)] = torch.tensor(fr)
    torch.save(tensor_en, output_file_en)
    torch.save(tensor_fr, output_file_fr)

    print(f"Saved tokenised sequences to {output_file_en} and {output_file_fr}")
    

class ParallelDataset:
    """
    Simple iterator:
    provides next() and reset() methods as LanguiniDatasetIterator
    but instead of (x, y, is_padded), next() returns ((x_l1, y_l1, is_padded_l1), (x_l2, y_l2, is_padded_l2))
    where x and y are of shape (bsz, seqlen) 
    """

    def __init__(self, batch_size, device, tokens_l1, tokens_l2):
        """
        Args:
            batch_size: batch size
            device: device
            tokens_l1: tensor (n_samples, seqlen) of tokenised sequences for language 1
            tokens_l2: tensor (n_samples, seqlen) of tokenised sequences for language 2
        """   
        assert len(tokens_l1) == len(tokens_l2)
        self.tokens_l1 = tokens_l1
        self.tokens_l2 = tokens_l2
        self.idx = 0
        self.device = device
        self.bsz = batch_size
    
    def __next__(self):
        if self.idx + self.bsz > len(self.tokens_l1):
            raise StopIteration()
        curr_slice = slice(self.idx, self.idx+self.bsz)
        # retrieve batch
        x_l1 = self.tokens_l1[curr_slice].to(self.device)
        x_l2 = self.tokens_l2[curr_slice].to(self.device)
        # create y
        y_l1, y_l2 = torch.zeros_like(x_l1), torch.zeros_like(x_l2)
        y_l1[:, :-1] = x_l1[:, 1:] # last token in y is 0 (padding)
        y_l2[:, :-1] = x_l2[:, 1:]

        self.idx += self.bsz
        return (x_l1, y_l1, True), (x_l2, y_l2, True)
    
    def reset(self):
        self.idx = 0
    

class ParallelEnFrDataset(ParallelDataset):
    def __init__(self, batch_size, device, combined_tokeniser, dataset_dir="data/fr-en"):
        monoling_vocab_size = combined_tokeniser.original_vocab_size
        tokens_en = torch.load(os.path.join(dataset_dir, f"tokenised_en_{monoling_vocab_size}.pt"))
        tokens_fr = torch.load(os.path.join(dataset_dir, f"tokenised_fr_{monoling_vocab_size}.pt"))
        tokens_en = combined_tokeniser.map_ids(tokens_en, is_l2=False)
        tokens_fr = combined_tokeniser.map_ids(tokens_fr, is_l2=True)
        super().__init__(batch_size, device, tokens_en, tokens_fr)


class ParallelEn1En2Dataset(ParallelDataset):
    """only works for fully duplicated vocab!"""
    def __init__(self, batch_size, device, original_vocab_size, dataset_dir="data/fr-en"):
        self.tokens_en_1 = torch.load(os.path.join(dataset_dir, f"tokenised_en_{original_vocab_size}.pt"))
        self.tokens_en_2 = torch.where(self.tokens_en_1 > 0, self.tokens_en_1 + original_vocab_size, 0) # 0 is padding
        super().__init__(batch_size, device, self.tokens_en_1, self.tokens_en_2)


if __name__ == "__main__":
    load_and_preprocess()
