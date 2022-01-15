# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy

import sentencepiece as spm
import fastBPE

import torch
from torch.distributions import Categorical
from torch.nn.functional import kl_div

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator


@dataclass
class ImitKDConfig(FairseqDataclass):
    expert: str = field(
        default="checkpoint_best.pt",
        metadata={"help": "NMT model to use as expert"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    beta: int = field(
        default=1,
        metadata={"help": "replacement prop"},
    )

    expert_vocab_tgt: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.de.txt",
        metadata={"help": "vocab for nmt model output"},
    )
    expert_vocab_src: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.en.txt",
        metadata={"help": "vocab for nmt model input"},
    )
    path: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/",
        metadata={"help": "directory with expert's dictionaries"},
    )
    sp_model: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/covost/en"
                "/spm_bpe8000_st_en_de.model",
        metadata={"help": "student's sentencepiece model"},
    )

    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )
    data_mix_rate: int = field(
        default=1,
        metadata={"help": "number of step to run before updating the model;s parameters"},
    )


def valid_loss(lprobs, target, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss


def imit_kd_loss(
        generated_dataset,
        model,
        expert,
        source_text,
        model_dict,
        expert_vocab_tgt,
        sp_model,
        bpe,
        ignore_index,
        source_lengths
):
    """
    encoded_prevs = []
    for s in generated_dataset["net_input"]["prev_output_tokens"]:
        encoded_prevs.append(model_dict.string(utils.strip_pad(s, model_dict.pad()),
                                               bpe_symbol='sentencepiece_fastBPE',
                                               escape_unk=True,
                                               include_eos=False
                                               )
                             )
    encoded_prevs = bpe.apply(encoded_prevs)
    """
    sample_expert = {
        "id": generated_dataset["id"],
        "net_input": {
            "src_tokens": source_text.cuda(),
            "src_lengths": source_lengths,
            "prev_output_tokens": generated_dataset["net_input"]["prev_output_tokens"].cuda()
        },
        "target": generated_dataset["target"],
        "target_lengths": generated_dataset["target_lengths"],
        "ntokens": generated_dataset["ntokens"],
        "nsentences": generated_dataset["nsentences"],
    }
    with torch.no_grad():
        expert_logits = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=True).detach()
        """
        for i, t in enumerate(expert_preds):
            print(i)
            print(expert_vocab_tgt.encode_line(encoded_prevs[i], add_if_not_exist=false, append_eos=true))
            print("student input :",
                  expert_vocab_tgt.string(
                      expert_vocab_tgt.encode_line(encoded_prevs[i], add_if_not_exist=false, append_eos=true),
                      bpe_symbol='fastbpe',
                      escape_unk=true,
                      include_eos=true
                  )
                  )
            print("expert prediction: ", expert_vocab_tgt.string(
                utils.strip_pad(t, expert_vocab_tgt.pad()), bpe_symbol='fastbpe', escape_unk=true, include_eos=true
            )
                  )
            print(utils.strip_pad(t, expert_vocab_tgt.pad()))
            print("target: ", model_dict.string(utils.strip_pad(generated_dataset["target"][i], model_dict.pad()),
                                                bpe_symbol='sentencepiece_fastBPE', escape_unk=True, include_eos=True))
            ep = expert_vocab_tgt.string(
                utils.strip_pad(t, expert_vocab_tgt.pad()), bpe_symbol='fastBPE', escape_unk=True, include_eos=True
            )
            print(" ".join(sp_model.EncodeAsPieces(ep)).replace("< / s >", "</s>").replace("<< unk >>", "<unk>"))
            print(model_dict.eos())
            print(model_dict.encode_line(
                " ".join(sp_model.EncodeAsPieces(ep)).replace("< / s >", "</s>").replace("<< unk >>", "<unk>"),
                add_if_not_exist=False,
                append_eos=True)
            )
            print(model_dict.string(
                model_dict.encode_line(
                    " ".join(sp_model.EncodeAsPieces(ep)).replace("< / s >", "</s>").replace("<< unk >>", "<unk>"),
                    add_if_not_exist=False,
                    append_eos=True)
            )
            )
        expert_preds_in_model_vocab = [
            model_dict.encode_line(
                " ".join(sp_model.EncodeAsPieces(
                    expert_vocab_tgt.string(
                        utils.strip_pad(t, expert_vocab_tgt.pad()), bpe_symbol='fastBPE', escape_unk=True,
                        include_eos=True
                    )
                )
                ).replace("< / s >", "</s>").replace("<< unk >>", "<unk>"),
                add_if_not_exist=False,
                )
            for t in expert_preds
        ]
        preds = collate_tokens(
            expert_preds_in_model_vocab,
            model_dict.pad(),
            model_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )
        preds = preds.to(torch.int64).cuda()
        """
    generated_dataset["net_input"].pop("src_text", None)
    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    # if preds.dim() == lprobs.dim() - 1:
        # preds = preds.unsqueeze(-1)
    # if preds.shape[1] > lprobs.shape[1]:
        # preds = preds[:, :lprobs.shape[1], :]
    #imit_kd_loss_for_sample = -lprobs.gather(dim=-1, index=preds)
    pad_mask = (lprobs == model_dict.pad())
    # pad_mask |= (lprobs == model_dict.eos())      inplace or: original paper removes eos from loss, fairseq generally keeps it - keep it for now
    lprobs = lprobs[~pad_mask]
    pad_mask = (expert_logits == expert_vocab_tgt.pad())
    expert_logits = expert_logits[~pad_mask]
    kl_loss = kl_div(lprobs, expert_logits, reduction="sum", log_target=True)
    # if ignore_index is not None:
        # pad_mask = preds.eq(ignore_index)
        # imit_kd_loss_for_sample.masked_fill_(pad_mask, 0.0)
    # imit_kd_loss_for_sample = imit_kd_loss_for_sample.sum()
    return kl_loss


@register_criterion(
    "imit_kd_kl_div", dataclass=ImitKDConfig
)
class ImitKD(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            beta,
            sp_model,
            bpe_codes,
            data_mix_rate,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.ignore_prefix_size = ignore_prefix_size
        self.data_mix_rate = data_mix_rate
        self.report_accuracy = report_accuracy
        self.expert, _ = load_model_ensemble([expert], arg_overrides={"data": path})
        self.expert = self.expert[-1]
        self.expert = self.expert.eval()
        self.expert_vocab_src = Dictionary.load(expert_vocab_src)
        self.expert_vocab_tgt = Dictionary.load(expert_vocab_tgt)
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.eos = self.dict.eos()
        self.bpe = fastBPE.fastBPE(bpe_codes, expert_vocab_tgt)
        self.pad_idx = self.padding_idx
        self.sentence_avg = False
        self.beta = beta
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model)
        self.sp_model.requires_grad = False

    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_s = copy.deepcopy(sample)
        sample_s["net_input"].pop("src_text", None)
        net_output = model(**sample_s["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True, valid=False):
        if valid:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.padding_idx, reduce=reduce)
        else:
            source_text, source_lengths = self.transform_source_tokens_into_expert_voc(sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            generated_dataset = self.generate_imit_batch(model, sample_s)
            loss = imit_kd_loss(
                generated_dataset,
                model,
                self.expert,
                source_text,
                self.dict,
                self.expert_vocab_tgt,
                self.sp_model,
                self.bpe,
                self.padding_idx,
                source_lengths
            )
        return loss

    def generate_imit_batch(self, student, sample):
        with torch.no_grad():
            student = student.eval()
            student_generator = SequenceGenerator([student], self.dict, beam_size=1)
            student_generator.cuda()
            hypos = student_generator._generate(sample)
            targets = sample["net_input"]["prev_output_tokens"].data.tolist()
            max_length = max([len(i) for i in targets])  # let's avoid blowing up the GPU RAM, shall we?
            dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
            samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            for i, hypo in enumerate(hypos):
                if samp_mask[i]:
                    targets[i] = hypo[0]["tokens"][:max_length].clone().detach()
                else:
                    targets[i] = torch.tensor(targets[i]).clone().detach()
            sample["net_input"]["prev_output_tokens"] = collate_tokens(
                targets,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).cuda()
            student.train()
        return sample

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        target = target.view(-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def transform_source_tokens_into_expert_voc(self, sample):
        source_text = sample["net_input"]["src_text"]
        source_texts = []
        for line in source_text:
            if isinstance(line, list):
                for text in line:
                    source_texts.append(self.expert_vocab_src.encode_line(
                        self.bpe.apply([text])[0],
                        add_if_not_exist=False,
                        append_eos=True)
                    )
            else:
                source_texts.append(self.expert_vocab_src.encode_line(
                    self.bpe.apply([line])[0],
                    add_if_not_exist=False,
                    append_eos=True)
                )
        src_lengths = [len(text) for text in source_texts]
        source_text = collate_tokens(
            source_texts,
            self.expert_vocab_src.pad(),
            self.expert_vocab_src.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )
        return source_text, src_lengths


def collate_tokens(values, pad_idx, eos, left_pad, move_eos_to_beginning):
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos
            dst[0] = eos
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            copy_tensor(v, res[i][size - len(v):])
        else:
            copy_tensor(v, res[i][:len(v)])
    return res
