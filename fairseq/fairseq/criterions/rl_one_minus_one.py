# R. H. loss function markings

import math
from difflib import SequenceMatcher

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def one_minus_one_nll(lprobs, target, rl_value_pos, rl_value_neg, ignore_index=None, reduce=True):
    predictions = lprobs.argmax(-1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        predictions = predictions.unsqueeze(-1)
    predicted_class_probs = -lprobs.gather(dim=-1, index=predictions)
    total_rewards = []
    target_list = target.tolist()
    pred_list = predictions.tolist()
    for p_list, t_list in zip(pred_list, target_list):
        if type(p_list[0]) == list:
            p_list_flat = [index for [index] in p_list]
        else:
            p_list_flat = p_list[:]
        if type(t_list[0]) == list:
            t_list_flat = [index for [index] in t_list]
        else:
            t_list_flat = t_list[:]
        s = SequenceMatcher(lambda x: x == ignore_index, p_list_flat, t_list_flat)
        rewards = [rl_value_neg for _ in p_list]
        pos_matches = s.get_matching_blocks()
        for match in pos_matches:
            if match.size > 0:
                pred_start = match.a
                end = pred_start + match.size
                rewards[pred_start:end] = [rl_value_pos for _ in range(match.size)]
        total_rewards.append(rewards)
        #print(sum(rewards)/len(rewards)) 
    nll_loss = -lprobs.gather(dim=-1, index=target)
    total_rewards = torch.tensor(total_rewards).cuda().unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        total_rewards.masked_fill_(pad_mask, 0.0)
        predicted_class_probs.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    loss = (total_rewards.squeeze(-1) * predicted_class_probs.squeeze(-1)).sum()
    return loss, nll_loss.sum().detach()


@register_criterion("rl_one_minus_one")
class RL_one_minus_one(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        rl_value_pos,
        rl_value_neg,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.rl_value_pos = rl_value_pos
        self.rl_value_neg = rl_value_neg

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--rl_value_pos', default=1., type=float, metavar='R',
                            help='positive value for rl')
        parser.add_argument('--rl_value_neg', default=-1., type=float, metavar='R',
                            help='positive value for rl')

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
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
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                raise ValueError(
                    "batch must be first"
                )
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = one_minus_one_nll(
            lprobs,
            target,
            self.rl_value_pos,
            self.rl_value_neg,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
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
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
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
