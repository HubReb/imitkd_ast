# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

import torch

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        ## knn related items
        parser.add_argument('--knn-keytype', type=str, default=None,
                            help='use last_ffn_input')
        parser.add_argument('--probe', default=8, type=int,
                            help='for FAISS, the number of lists to query')
        parser.add_argument('--k', default=1024, type=int,
                            help='number of nearest neighbors to retrieve')
        # default value is the one for news-comm-14
        parser.add_argument('--dstore-size', default=9651607, type=int,
                            help='number of items in the knn datastore')
        parser.add_argument('--dstore-filename', type=str, default=None,
                            help='File where the knn datastore is saved')
        parser.add_argument('--indexfile', type=str, default=None,
                            help='File containing the index built using faiss for knn')
        parser.add_argument('--lmbda', default=0.0, type=float,
                            help='controls interpolation with knn, 0.0 = no knn')
        parser.add_argument('--knn-sim-func', default=None, type=str,
                            help='similarity function to use for knns')
        parser.add_argument('--faiss-metric-type', default='l2', type=str,
                            help='the distance metric for faiss')
        parser.add_argument('--no-load-keys', default=False, action='store_true',
                            help='do not load keys')
        parser.add_argument('--dstore-fp16', default=False, action='store_true',
                            help='if true, datastore items are saved in fp16 and int16')
        parser.add_argument('--move-dstore-to-mem', default=False, action='store_true',
                            help='move the keys and values for knn to memory')
        ## knnlm related items
    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.tgt_dict

    @property
    def dummy_vocab(self):
        return True

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        task_extra_args=None
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs, task_extra_args=task_extra_args
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if str(type(criterion)).endswith("OracleForcedDecoding'>"):
                loss, sample_size, logging_output = criterion(model, sample, valid=True)
            elif str(type(criterion)).endswith("OracleForcedDecodingMNT'>"):
                loss, sample_size, logging_output = criterion(model, sample, valid=True)
            elif str(type(criterion)).endswith("OracleDiff'>"):
                loss, sample_size, logging_output = criterion(model, sample, valid=True)
            elif str(type(criterion)).endswith("Difference'>"):
                loss, sample_size, logging_output = criterion(model, sample, valid=True)
            elif str(type(criterion)).endswith("ImitKD'>"):
                loss, sample_size, logging_output = criterion(model, sample, valid=True)

            else:
                loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
