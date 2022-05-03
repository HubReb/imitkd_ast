#!/usr/bin/env python3

from typing import List, Tuple
from collections import defaultdict

from sacrebleu.metrics.lib_ter import translation_edit_rate


def calculate_ter_score(hypothesis: str, reference: str) -> float:
    """
    Calculate translation edit rate for one hypothesis to one reference string.

    :param hypothesis: tokenized hypothesis string
    :param reference: tokenized hypothesis string
    :return: translation edit rate (TER) in percentage
    """
    hypothesis_list = hypothesis.split()
    reference_list = reference.split()
    best_num_edits, ref_length = translation_edit_rate(hypothesis_list, reference_list)
    ter_score = best_num_edits / ref_length * 100
    return ter_score


def calculate_ter_score_for_each_sentence_in_data(hypotheses: List[str], references: List[str]) -> List[
        Tuple[float, str, str]]:
    """
    Calculate TER score for each hypothesis, reference pair available and rank each pair with decreasing TER score.

    :param hypotheses: list of tokenized hypothesis strings
    :param references: list of tokenized reference strings
    :return:  return elements sorted by decreasing TER score
    """
    ter_scores = defaultdict(tuple)
    for hypothesis, reference in zip(hypotheses, references):
        ter_score = calculate_ter_score(hypothesis, reference)
        ter_scores[ter_score] = (hypothesis, reference)
    ranked_ter_score_list = sorted(
        [(ter_score, hypothesis, reference) for ter_score, (hypothesis, reference) in ter_scores.items()],
        key=lambda x: x[0],
        reverse=True)
    return ranked_ter_score_list


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


def compare_ter_scores(
        ter_score_list_one: List[Tuple[float, str, str]],
        ter_score_list_two: List[Tuple[float, str, str]],
        model1: str, model2: str
) -> str:
    comparisons = []
    for (ter_score, hypo, ref) in ter_score_list_one:
        for (ter_score_other, other_hypo, other_ref) in ter_score_list_two:
            # we need to remove those to get ANYTHING OTHER than TO REMOVE and PLEASE REMOVE as highest TER samples
            if "TO REMOVE" in ref or "PLEASE REMOVE" in ref:
                continue
            if ref == other_ref:
                comparisons.append((ter_score - ter_score_other, ter_score, ter_score_other, hypo, other_hypo, ref))
    comparisons.sort(key=lambda x: x[0], reverse=True)
    markdown_string = ""
    for ter_difference, first_score, second_score, first_hypo, second_hypo, ref_string in comparisons[:15]:
        string = f"_TER Difference_: {ter_difference}\n_TER {model1}_: {first_score}\n_TER {model2}_: {second_score}\n" + \
            f"*Reference*:\n{ref_string}\n*{model1} hypo*: {first_hypo}\n*{model2} hypo*: {second_hypo}"
        markdown_string += string
        markdown_string += "\n" + "---" * 15 + "\n"
    return markdown_string


def save_to_json(ter_content, filename):
    with open(f"{filename}.md", "w") as f:
        f.write(ter_content)


# pytest tests
def test_ter_all_correct():
    assert 0 == calculate_ter_score("Hello , goodbye", "Hello , goodbye")


def test_ter_all_wrong():
    assert 100 == calculate_ter_score("Hello , goodbye", "Goodbye ! or")


def test_calculate_ter_score_for_each_sentence_in_data():
    assert [(100, "Hello , goodbye", "Goodbye ! or"), (0, "Hello , goodbye",  "Hello , goodbye")] == \
           calculate_ter_score_for_each_sentence_in_data(
               ["Hello , goodbye", "Hello , goodbye"], ["Hello , goodbye", "Goodbye ! or"]
           )


def test_get_reference_and_hypothesis_strings_from_datafile():
    assert ["Hello , goodbye", "Goodbye ! or"], ["Hello , goodbye",  "Hello , goodbye"] == \
           get_reference_and_hypothesis_strings_from_datafile("test_file.txt")


def test_combination():
    r, h = get_reference_and_hypothesis_strings_from_datafile("test_file.txt")
    assert [(100, "Hello , goodbye", "Goodbye ! or"), (0, "Hello , goodbye",  "Hello , goodbye")] == \
        calculate_ter_score_for_each_sentence_in_data(h, r)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file_one", help="name of fairseq-generate log file")
    parser.add_argument("log_file_two", help="name of fairseq-generate log file")
    parser.add_argument("model_one_name", help="specify name of first model")
    parser.add_argument("model_two_name", help="specify name of second model")
    args = parser.parse_args()
    fairseq_file = args.log_file_one
    fairseq_file_for_comparison = args.log_file_two
    result_filename = f"ter_comparison_{fairseq_file.split('.txt')[0]}_{fairseq_file_for_comparison.split('.txt')[0]}"
    references_list, hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(fairseq_file)
    ter_score_examples = calculate_ter_score_for_each_sentence_in_data(hypotheses_list, references_list)
    comparison_references_list, comparison_hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(
        fairseq_file_for_comparison
    )
    comparison_ter_score_examples = calculate_ter_score_for_each_sentence_in_data(
        comparison_hypotheses_list,
        comparison_references_list
    )
    result_string = compare_ter_scores(
        ter_score_examples,
        comparison_ter_score_examples,
        args.model_one_name,
        args.model_two_name
    )
    save_to_json(result_string, result_filename)
