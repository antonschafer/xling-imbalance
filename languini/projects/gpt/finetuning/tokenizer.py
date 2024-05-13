from typing import Any
import torch

from languini.train_lib import train_utils

# wrapper around sentencepiece tokenizer to be compatible with huggingface tokenizers
class HFTokenizer:
    def __init__(self, config):
        self.sp = train_utils.load_tokeniser(name=config.dataset)
        self.model_max_length = config.seq_len # TODO

        # use end of doc token as eos token
        _eos_id = self.sp.encode("<D>")
        assert len(_eos_id) == 1
        self.eos_id = _eos_id[0]
    
    def save_pretrained(self, output_dir):
        # ignore, not needed, just need config
        pass

    def _encode_sentences(self, sentences):
        """encode sentences and add eos token"""
        return [ids + [self.eos_id] for ids in self.sp.encode(sentences)]

    def __call__(self, text1, text2=None, *, padding=True, truncation=True, max_length=None): #, return_tensors=None, **kwargs):
        """emulate huggingface tokenizer"""
        assert not isinstance(text1, str), "text must be iterable of strings, single string not implemented"

        # encode
        ids = self._encode_sentences(text1)

        # add second sentence if given
        if text2 is not None:
            assert len(text1) == len(text2)
            ids = [x + y for x, y in zip(ids, self._encode_sentences(text2))]

        if padding:
            if padding == "max_length":
                ids = [x + [0] * (max_length - len(x)) for x in ids]
            else:
                raise NotImplementedError()

        if truncation:
            ids = [x[:max_length] for x in ids]

        return {"input_ids": ids}
