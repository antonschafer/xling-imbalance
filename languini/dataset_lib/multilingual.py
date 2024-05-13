"""Datasets for multilingual data that follow the same interface as LanguiniDatasetIterator"""


import random
import torch
from languini.common_lib.parallel_utils import mprint
from languini.dataset_lib.languini_books import LanguiniDatasetIterator


class LanguageScheduler:
    """
    Changes p_clone/p_l2 over time according to a schedule of evenly spaced steps.
    """
    def __init__(self, schedule, n_total_steps, datasets, attr_name):
        assert len(schedule) > 1
        self.n_total_steps = n_total_steps
        self.schedule = [float(x) for x in schedule.split("_")]
        self.attr_name = attr_name
        self.datasets = datasets
        self.curr_step = 0
        self.idx = 0
        self.episode_length = n_total_steps // len(self.schedule)

        # check that first value matches
        for ds in datasets:
            if getattr(ds, attr_name) != self.schedule[0]:
                raise ValueError(f"Initial value of {attr_name} does not match schedule.")
    
    def step(self):
        """
        To be called at the end of each training step.
        """
        assert self.curr_step < self.n_total_steps
        self.curr_step += 1
        if self.curr_step % self.episode_length == 0 and self.idx < len(self.schedule) - 1:
            self.idx += 1
            for ds in self.datasets:
                setattr(ds, self.attr_name, self.schedule[self.idx])
            mprint(f"language scheduler -- switched to {self.schedule[self.idx]} for {self.attr_name} after {self.curr_step} steps.")


class ClonedLanguageDataset(LanguiniDatasetIterator):
    def __init__(self, *, num_languages, p_clone, frac_clone, sp, **kwargs):
        """
        Initialise the dataset iterator. 

        Args:
            same as LanguiniDatasetIterator. Additionally:
            num_languages (int): number of languages to clone the dataset into.
            p_clone (float): probability of using a cloned language for a sample.
            frac_clone (float): fraction of the vocabulary to clone.
            sp: SentencePiece tokenizer.
        """
        assert num_languages > 1
        assert frac_clone == 1.0 or num_languages == 2, "not supported"
        super().__init__(**kwargs, sp=sp)
        self.num_languages = num_languages
        self.p_clone = p_clone
        self.frac_clone = frac_clone
        self.sp = sp
        self.original_vocab_size = self.sp.vocab_size()

        # randomly select a fraction of subword ids that are cloned
        clone_seed = 0
        self.n_cloned = int(frac_clone * self.original_vocab_size)
        self.is_cloned = torch.zeros(self.original_vocab_size, dtype=torch.bool, device=self.device)
        all_ids = list(range(self.original_vocab_size))
        random.Random(clone_seed).shuffle(all_ids)
        for i in all_ids[:self.n_cloned]:
            self.is_cloned[i] = True
        
        # randomly select a fraction of subwords that are independent on sequence langauge
        # currently not supported
        self.frac_independent = 0.0 
        self.is_independent = torch.zeros_like(self.is_cloned)

        self.vocab_size = self.original_vocab_size * self.num_languages

    def __next__(self):
        seq, is_padded = super().__next__(return_seq=True)

        micro_batches, micro_bsz, seqlen_plus1 = seq.shape
        bsz = micro_batches * micro_bsz

        # sample which cloned language to use for each sequence
        cloned_lang = torch.randint(1, self.num_languages, (bsz, 1), device=self.device)
        do_clone_seq = torch.rand(bsz, 1, dtype=torch.float, device=self.device) < self.p_clone
        lang = torch.where(do_clone_seq, cloned_lang, 0)
        # repeat for each token in the sequence
        lang = lang.view(micro_batches, micro_bsz, 1).expand(seq.shape)

        # resample for the tokens that are independent of the sequence language (only supported for 2 languages)
        assert self.num_languages == 2 or self.frac_independent == 0, "not supported"
        do_clone_tok = torch.rand_like(seq, dtype=torch.float) < self.p_clone
        lang = torch.where(self.is_independent[seq], do_clone_tok, lang)

        # lang i has ids [i * size, (i+1) * size)
        lang_offset = lang * self.original_vocab_size
        # map to vocabulary of the "language", don't map padding (0), only map cloned subwords
        seq = torch.where((seq > 0) & self.is_cloned[seq], seq + lang_offset, seq)

        batch_x = seq[:, :, :-1]
        batch_y = seq[:, :, 1:]
        print(batch_x)
        print(batch_y)
        return batch_x, batch_y, is_padded

    def decode(self, ids):
        assert ids.ndim == 1
        # map to original vocab
        ids = ids % self.original_vocab_size
        return self.sp.decode(ids.cpu().tolist())


class CombinedTokenizer:
    """
    Combines to existing tokenisers into one, merging the vocabularies if specified.

    (this is for the purpose of the paper. In general, one should probably train a single tokeniser on the combined data.)
    """
    def __init__(self, sp1, sp2, merge_vocab, verbose=False):
        if sp1.vocab_size() != sp2.vocab_size():
            raise NotImplementedError("Vocabularies must have the same size.")
        self.original_vocab_size = sp1.vocab_size()
        self.sp1 = sp1
        self.sp2 = sp2

        # create the combined vocabulary
        if merge_vocab:
            self.sp2_id_to_combined_id = torch.full((self.original_vocab_size,), -1, dtype=torch.long)
            sp1_piece_to_id = {self.sp1.id_to_piece(i): i for i in range(self.original_vocab_size)}
            n_added = 0
            for i in range(self.original_vocab_size):
                piece = self.sp2.id_to_piece(i)
                if piece in sp1_piece_to_id:
                    # piece exists in both vocabularies, merge
                    self.sp2_id_to_combined_id[i] = sp1_piece_to_id[piece]
                else:
                    # create a new id
                    combined_id = self.original_vocab_size + n_added
                    n_added += 1
                    self.sp2_id_to_combined_id[i] = combined_id
            self.combined_vocab_size = self.original_vocab_size + n_added
        else:
            self.combined_vocab_size = 2 * self.original_vocab_size
            self.sp2_id_to_combined_id = torch.arange(self.original_vocab_size, 2 * self.original_vocab_size)
        
        # make sure that padding does not get remapped, stays 0
        self.sp2_id_to_combined_id[0] = 0
        
        # track the reverse mapping
        self.combined_id_to_sp2_id = {int(combined_id): sp2_id for sp2_id, combined_id in enumerate(self.sp2_id_to_combined_id)}
        assert len(self.combined_id_to_sp2_id) == self.original_vocab_size
        assert max(self.combined_id_to_sp2_id.keys()) == self.combined_vocab_size - 1

        if verbose:
            mprint(f"Combined two tokenisers {'with' if merge_vocab else 'without'} merging.")
            mprint(f"\toriginal vocab size: 2 * {self.original_vocab_size}, size of combined vocab: {self.combined_vocab_size}, num merged: {2 * self.original_vocab_size - self.combined_vocab_size} (padding always merged)")
    
    def map_ids(self, ids, is_l2):
        """
        Map ids generated by either the first or the second tokeniser to the combined vocabulary.
        """
        if is_l2:
            return self.sp2_id_to_combined_id[ids]
        else:
            return ids
        
    def encode(self, text):
        raise NotImplementedError()
    
    def decode(self, ids):
        assert ids.ndim == 1
        if max(ids) < self.original_vocab_size:
            # all ids are from the first tokeniser, can decode directly
            return self.sp1.decode(ids.cpu().tolist())
        else:
            # some ids are from the second tokeniser, map all to second tokeniser, then decode
            return self.sp2.decode([self.combined_id_to_sp2_id[int(x)] for x in ids])


class BilingualDataset:
    def __init__(
            self,
            *,
            data_path,
            data_path_2,
            sp,
            sp2,
            merge_vocab,
            p_l2,
            verbose=False,
            **kwargs
        ):
        """
        Initialise the dataset iterator. 

        Args:
            same as LanguiniDatasetIterator. Additionally:
            data_path (str): path to the first dataset.
            data_path_2 (str): path to the second dataset.
            sp: SentencePiece tokenizer for the first language.
            sp_2: SentencePiece tokenizer for the second language.
            merge_vocab (bool): whether to merge the vocabularies.
            p_l2 (float): probability of using the second language for a sample.
            verbose (bool): whether to print information about the merged vocabularies.
        """
        self.device = kwargs["device"]
        self.seq_len = kwargs["sequence_length"]
        self.batch_idxs = kwargs["batch_idxs"]
        self.micro_batches = kwargs["micro_batches"]
        self.bsz = len(self.batch_idxs)
        kwargs["device"] = "cpu" # avoid cluttering the GPU as we have to buffer data
        self.ds1 = LanguiniDatasetIterator(data_path=data_path, sp=sp, **kwargs)
        self.ds2 = LanguiniDatasetIterator(data_path=data_path_2, sp=sp2, **kwargs)
        self.merge_vocab = merge_vocab
        self.p_l2 = p_l2
        self.lang_seed = 0
        self.combined_tokeniser = CombinedTokenizer(sp, sp2, merge_vocab, verbose=verbose)

        # initialize buffers
        self.reset()

        self.vocab_size = self.combined_tokeniser.combined_vocab_size
    
    def __iter__(self):
        return self

    def decode(self, ids):
        return self.combined_tokeniser.decode(ids)

    def _fill_buffer(self, use_ds1, n=1):
        dataset = self.ds1 if use_ds1 else self.ds2
        buffer = self.buffer_1 if use_ds1 else self.buffer_2

        for _ in range(n):
            batch_x, batch_y, is_padded = next(dataset)

            batch_x = batch_x.view(-1, self.seq_len)
            batch_y = batch_y.view(-1, self.seq_len)

            for i, idx in enumerate(self.batch_idxs):
                buffer[idx].append((batch_x[i], batch_y[i], is_padded))
    
    def __next__(self):
        # we need this complex logic because we want to sample the language per sequence, not per batch
        batch_use_ds1 = torch.rand(len(self.batch_idxs), generator=self.torch_rng) > self.p_l2
        batch_x, batch_y, is_padded = [], [], []
        for i, use_ds1 in enumerate(batch_use_ds1):
            buffer = self.buffer_1 if use_ds1 else self.buffer_2
            if len(buffer[i]) == 0:
                self._fill_buffer(use_ds1, n=4)
            assert len(buffer[i]) > 0
            x, y, p = buffer[i].pop()

            x = self.combined_tokeniser.map_ids(x, not use_ds1)
            y = self.combined_tokeniser.map_ids(y, not use_ds1)

            batch_x.append(x)
            batch_y.append(y)
            is_padded.append(p)
        
        batch_x = torch.stack(batch_x).view(self.micro_batches, self.bsz // self.micro_batches, self.seq_len)
        batch_y = torch.stack(batch_y).view(self.micro_batches, self.bsz // self.micro_batches, self.seq_len)
        is_padded = any(is_padded)

        return batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True), is_padded

    def reset(self):
        # reset state
        self.buffer_1 = [[] for _ in self.batch_idxs]
        self.buffer_2 = [[] for _ in self.batch_idxs]
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(self.lang_seed)
        self.ds1.reset()
        self.ds2.reset()
