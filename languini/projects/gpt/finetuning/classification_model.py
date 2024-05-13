# wraps languini models in huggingface

import os
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from languini.common_lib import experiment_utils
from languini.projects.gpt.model import Model

class HFClassificationModelConfig(PretrainedConfig):
    def __init__(
            self,
            *,
            num_labels,
            clf_head_dropout_rate,
            L_train,
            wandb_run=None,
            base_config=None,
            device=None,
            problem_type=None,
        ):
        assert (wandb_run is None) != (base_config is None)
        assert L_train in ["L1", "L2"]

        if wandb_run is not None:
            config_file = experiment_utils.load_wandb_file(wandb_run, "config.pickle")
            with open(config_file, "rb") as f:
                base_config = pickle.load(f)

        self.base_wandb_run = wandb_run
        self.base_config = base_config
        self.num_labels = num_labels
        self.clf_head_dropout_rate = clf_head_dropout_rate
        self.problem_type = problem_type
        self.L_train = L_train
        
        for k, v in base_config.items():
            setattr(self, k, v)
        
        if device is not None:
            # overwrite device from base config
            self.device = device
            self.base_config.device = device
    
    def to_json_string(self, use_diff=True) -> str:
        # ignore use_diff
        relevant_fields = {k: v for k, v in self.__dict__.items() if k not in self.base_config and k != "base_config"}
        return json.dumps(relevant_fields, indent=2, sort_keys=True) + "\n"


class HFClassificationModel(PreTrainedModel):
    config_class = HFClassificationModelConfig

    @classmethod
    def from_pretrained(cls, config: HFClassificationModelConfig):
        return cls(config)

    def __init__(self, config: HFClassificationModelConfig):
        super().__init__(config)
        c = config

        # language config
        self.L_train = c.L_train
        self.L_predict = c.L_train # use same by default
        assert c.frac_clone in [0, 1], "not implemented"
        assert not c.data_root_2, "not implemented"
        assert not (c.frac_clone == 0 and c.L_train == "L2"), "not implemented"

        # load state from wandb
        checkpoint_file, config_file = experiment_utils.load_wandb_checkpoint_and_config(c.base_wandb_run)
        with open(config_file, "rb") as f:
            _languini_base_config = pickle.load(f)
        # check that config matches
        for k, _v in _languini_base_config.items():
            if k == "device": continue
            v = getattr(config, k)
            if torch.is_tensor(v):
                assert torch.equal(v, _v), f"Config mismatch: {k}={_v} != {v}"
            elif isinstance(v, np.ndarray):
                assert np.equal(v, _v), f"Config mismatch: {k}={_v} != {v}"
            else:
                assert v == _v, f"Config mismatch: {k}={_v} != {v}"
        self.languini_base_model = Model(config)

        # load checkpoint
        self.load_base_model_from_checkpoint(checkpoint_file)

        # classification head
        self.dropout = nn.Dropout(c.clf_head_dropout_rate)
        self.classifier = nn.Linear(c.h_dim, c.num_labels) # don't change name, tuning script relies on it

    def load_base_model_from_checkpoint(self, checkpoint_file):
        """Loads the base model from a checkpoint on disk. Needed to map the state dict correctly"""
        new_state_dict = {}
        with open(checkpoint_file, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device)
            model_state_dict = checkpoint["model_state_dict"]
            for key, value in model_state_dict.items():
                new_state_dict[key.replace("module._orig_mod.", "")] = value
            self.languini_base_model.load_state_dict(new_state_dict)
    
    def set_prediction_language(self, L_predict):
        assert L_predict in ["L1", "L2"]
        if self.config.frac_clone == 0:
            raise ValueError("Cannot predict in cloned language if model wasn't trained on cloned language")
        self.L_predict = L_predict

    def to_language(self, input_ids):
        if self.training:
            L = self.L_train
        else:
            L = self.L_predict
        
        if L == "L1":
            return input_ids
        elif L == "L2":
            # TODO solve cleaner
            assert self.config.frac_clone == 1
            original_vocab_size = self.config.vocab_size // 2
            return torch.where(input_ids > 0, input_ids + original_vocab_size, 0)
        else:
            raise ValueError(f"Unknown language {L}")
        
    def forward(self, input_ids, labels=None, return_dict=None):
        c = self.config

        # map to correct language
        input_ids = self.to_language(input_ids)

        # run base model
        _, _, hidden_states = self.languini_base_model(input_ids, state=None, log=None, return_hidden=True)

        # classification head on last hidden state of last layer
        logits = self.classifier(self.dropout(hidden_states[-1, :, -1, :]))

        # compute loss, adpated from BertForSequenceClassification
        loss = None
        if labels is not None:
            if c.problem_type is None:
                if c.num_labels == 1:
                    c.problem_type = "regression"
                elif c.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    c.problem_type = "single_label_classification"
                else:
                    c.problem_type = "multi_label_classification"

            if c.problem_type == "regression":
                loss = F.mse_loss(logits.view(-1, c.num_labels), labels.view(-1))
            elif c.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, c.num_labels), labels.view(-1))
            elif c.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


