#!/usr/bin/env python3

from typing import List, Tuple
from collections import defaultdict

import pandas as pd
from sacrebleu.metrics.chrf import CHRF
import pandas as pd


def calculate_chrf_score(hypothesis: str, reference: str) -> float:
    """
    Calculate translation edit rate for one hypothesis to one reference string.

    :param hypothesis: tokenized hypothesis string
    :param reference: tokenized hypothesis string
    :return: translation edit rate (chrF) in percentage
    """
    hypothesis_list = hypothesis.split()
    reference_list = reference.split()
    best_num_edits, ref_length = translation_edit_rate(hypothesis_list, reference_list)
    chrf_score = best_num_edits / ref_length * 100
    return chrf_score


def calculate_chrf_score_for_each_sentence_in_data(hypotheses: List[str], references: List[str], ast_dataset_filename: str, asr_refs: List[str], asr_hypos: List[str]) -> List[
        Tuple[float, str, str, str, str]]:
    """
    Calculate chrF score for each hypothesis, reference pair available and rank each pair with decreasing chrF score.

    :param hypotheses: list of tokenized hypothesis strings
    :param references: list of tokenized reference strings
    :return:  return elements sorted by decreasing chrF score
    """
    chrf_scores = defaultdict(tuple)
    ast_dataset = pd.read_csv(ast_dataset_filename, sep="\t").filter(['tgt_text', 'src_text'], axis=1)
    reference_to_transcript = dict(ast_dataset.values)
    asr_refs_to_hypos = {}
    for r, h in zip(asr_refs, asr_hypos):
        asr_refs_to_hypos[r.strip()] = h
    chrfs = CHRF()
    for hypothesis, reference in zip(hypotheses, references):
        asr_transcript = reference_to_transcript[reference]
        try:
            asr_hypo = asr_refs_to_hypos[asr_transcript.strip()]
        except KeyError:
            print(asr_transcript)
        # chrf_score = calculate_chrf_score(hypothesis, reference)
        asr_transcript = reference_to_transcript[reference]
        score = chrfs.sentence_score(hypothesis, [reference]).score
        chrf_scores[score] = (hypothesis, reference, asr_transcript, asr_hypo)
    ranked_chrf_score_list = sorted(
        [(chrf_score, hypothesis, reference, asr_transcript, asr_hypo) for chrf_score, (hypothesis, reference, asr_transcript, asr_hypo) in chrf_scores.items()],
        key=lambda x: x[0],
        reverse=True)
    return ranked_chrf_score_list


def get_reference_and_hypothesis_strings_from_datafile(filename: str) -> Tuple[List[str], List[str]]:
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    references, hypotheses = [], []
    for line in data:
        if line.startswith("T-"):
            references.append(" ".join(line.split()[1:]))
        elif line.startswith("D-"):
            hypotheses.append(" ".join(line.split()[2:]))
    assert len(hypotheses) == len(references)
    return references, hypotheses


def compare_chrf_scores(
        chrf_score_list_one: List[Tuple[float, str, str]],
        chrf_score_list_two: List[Tuple[float, str, str]],
        model1: str, model2: str
) -> str:
    comparisons = []
    for (chrf_score, hypo, ref, transcript, _) in chrf_score_list_one:
        for (chrf_score_other, other_hypo, other_ref, other_transcript, asr_hypo) in chrf_score_list_two:
            # we need to remove those to get ANYTHING OTHER than TO REMOVE and PLEASE REMOVE as highest chrF samples
            if "TO REMOVE" in ref or "PLEASE REMOVE" in ref:
                continue
            if ref == other_ref:
                comparisons.append((chrf_score - chrf_score_other, chrf_score, chrf_score_other, hypo, other_hypo, ref, transcript, asr_hypo))
    comparisons.sort(key=lambda x: x[0], reverse=True)
    markdown_string = ""
    for ter_difference, first_score, second_score, first_hypo, second_hypo, ref_string, transcript, asr_hypo in comparisons[:15]:
        string = f"_chrF Difference_: {ter_difference}\n_chrF {model1}_: {first_score}\n_chrF {model2}_: {second_score}\n" + \
                 f"*Transcript*:\n{transcript}\n*ASR Output*:\n{asr_hypo}\n*Reference*:\n{ref_string}\n" + \
                 f"*{model1} hypo*: {first_hypo}\n*{model2} hypo*: {second_hypo}"
        markdown_string += string
        markdown_string += "\n" + "---" * 15 + "\n"
    return markdown_string


def save_to_json(ter_content, filename):
    with open(f"{filename}.md", "w") as f:
        f.write(ter_content)


# pytest tests
def test_ter_all_correct():
    assert 0 == calculate_chrf_score("Hello , goodbye", "Hello , goodbye")


def test_ter_all_wrong():
    assert 100 == calculate_chrf_score("Hello , goodbye", "Goodbye ! or")


def test_calculate_chrf_score_for_each_sentence_in_data():
    assert [(100, "Hello , goodbye", "Goodbye ! or"), (0, "Hello , goodbye",  "Hello , goodbye")] == \
           calculate_chrf_score_for_each_sentence_in_data(
               ["Hello , goodbye", "Hello , goodbye"], ["Hello , goodbye", "Goodbye ! or"]
           )


def test_get_reference_and_hypothesis_strings_from_datafile():
    assert ["Hello , goodbye", "Goodbye ! or"], ["Hello , goodbye",  "Hello , goodbye"] == \
           get_reference_and_hypothesis_strings_from_datafile("test_file.txt")


def test_combination():
    r, h = get_reference_and_hypothesis_strings_from_datafile("test_file.txt")
    assert [(100, "Hello , goodbye", "Goodbye ! or"), (0, "Hello , goodbye",  "Hello , goodbye")] == \
        calculate_chrf_score_for_each_sentence_in_data(h, r)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file_one", help="name of fairseq-generate log file")
    parser.add_argument("log_file_two", help="name of fairseq-generate log file")
    parser.add_argument("ast_data_set", help="tsv file of dataset")
    parser.add_argument("asr_output", help="name of fairseq-generate log file for asr model")
    parser.add_argument("model_one_name", help="specify name of first model")
    parser.add_argument("model_two_name", help="specify name of second model")
    args = parser.parse_args()
    fairseq_file = args.log_file_one
    fairseq_file_for_comparison = args.log_file_two
    result_filename = f"chrF_comparison_{fairseq_file.split('.txt')[0]}_{fairseq_file_for_comparison.split('.txt')[0]}"
    references_list, hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(fairseq_file)
    asr_ref, asr_hypos = get_reference_and_hypothesis_strings_from_datafile(args.asr_output)
    chrf_score_examples = calculate_chrf_score_for_each_sentence_in_data(hypotheses_list, references_list, args.ast_data_set, asr_ref, asr_hypos)
    comparison_references_list, comparison_hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(
        fairseq_file_for_comparison
    )
    comparison_chrf_score_examples = calculate_chrf_score_for_each_sentence_in_data(
        comparison_hypotheses_list,
        comparison_references_list,
        args.ast_data_set,
        asr_ref,
        asr_hypos
    )
    result_string = compare_chrf_scores(
        chrf_score_examples,
        comparison_chrf_score_examples,
        args.model_one_name,
        args.model_two_name
    )
    save_to_json(result_string, result_filename)
