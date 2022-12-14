#!/usr/bin/env python3

from typing import List, Tuple
from collections import defaultdict

from sacrebleu.metrics.lib_ter import translation_edit_rate
import pandas as pd


def calculate_ter_score(hypothesis: str, reference: str) -> float:
    """
    Calculates translation edit rate for one hypothesis to one reference string.

    Args:
        hypothesis: tokenized hypothesis string
        reference: tokenized hypothesis string
    Returns:
        translation edit rate (TER) in percentage
    """
    hypothesis_list = hypothesis.split()
    reference_list = reference.split()
    best_num_edits, ref_length = translation_edit_rate(hypothesis_list, reference_list)
    ter_score = best_num_edits / ref_length * 100
    return ter_score


def calculate_ter_score_for_each_sentence_in_data(hypotheses: List[str], references: List[str], ast_dataset_filename: str, asr_refs: List[str], generated_hypos_asr: List[str]) -> List[
        Tuple[float, str, str, str, str]]:
    """
    Calculates TER score for each hypothesis, reference pair available and rank each pair with decreasing TER score.
    
    Args:
        hypotheses: list of tokenized hypothesis strings
        references: list of tokenized reference strings
        ast_dataset_filename: name of csv containing AST dataset
        asr_refs: list of gold transcripts
        generated_hypos_asr: list of hypotheses generated by ASR model
    Returns:
        return elements sorted by decreasing TER score
    """
    ast_dataset = pd.read_csv(ast_dataset_filename, sep="\t").filter(['tgt_text', 'src_text'], axis=1)
    reference_to_transcript = dict(ast_dataset.values)
    asr_refs_to_hypos = {}
    for r, h in zip(asr_refs, generated_hypos_asr):
        asr_refs_to_hypos[r.strip()] = h
    ter_scores = defaultdict(tuple)
    for hypothesis, reference in zip(hypotheses, references):
        try:
            asr_transcript = reference_to_transcript[reference]
            asr_hypo = asr_refs_to_hypos[asr_transcript.strip()]
        except KeyError:
            continue
        ter_score = calculate_ter_score(hypothesis, reference)
        ter_scores[ter_score] = (hypothesis, reference, asr_transcript, asr_hypo)
    ranked_ter_score_list = sorted(
        [(ter_score, hypothesis, reference, asr_transcript, asr_hypo) for ter_score, (hypothesis, reference, asr_transcript, asr_hypo) in ter_scores.items()],
        key=lambda x: x[0],
        reverse=True)
    return ranked_ter_score_list


def get_reference_and_hypothesis_strings_from_datafile(filename: str) -> Tuple[List[str], List[str]]:
    """
    Extracts hypotheses and references from log file  of fairseq-generate

    Args:
        filename:  name of log file
    Returns: 
        references: All references in log file in a list
        hypotheses: All hypotheses in log file in a list
    """
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
        ter_score_list_one: List[Tuple[float, str, str, str, str]],
        ter_score_list_two: List[Tuple[float, str, str, str, str]],
        model1: str, model2: str
) -> str:
    """
    Extracts 15 samples with the highest difference of TER scores of model 1 and TER scores model 2)

    Args:
        ter_score_list_one: list of tuples of form (TER, hypothesis, reference, gold transcript, generated transcript)  for model 1
        ter_score_list_two: list of of same form as ter_score_list_one for model 2
        model1: name of the first model
        model2: name of the second model
    Returns:
        string containing hypotheses
        references
        TER scores
    """
    comparisons = []
    for (ter_score, hypo, ref, transcript, _) in ter_score_list_one:
        for (ter_score_other, other_hypo, other_ref, other_transcript, asr_hypo) in ter_score_list_two:
            # we need to remove those to get ANYTHING OTHER than TO REMOVE and PLEASE REMOVE as highest TER samples
            if "TO REMOVE" in ref or "PLEASE REMOVE" in ref or "xxx" == ref:        # processing went wrong on some samples - unk
                continue
            if ref == other_ref:
                if transcript != other_transcript:
                    print("SOMETHING WENT VERY WRONG HERE")
                comparisons.append((ter_score - ter_score_other, ter_score, ter_score_other, hypo, other_hypo, ref, transcript, asr_hypo))
    comparisons.sort(key=lambda x: x[0], reverse=True)
    markdown_string = ""
    for ter_difference, first_score, second_score, first_hypo, second_hypo, ref_string, transcript, asr_hypo in comparisons[:15]:
        string = f"_TER Difference_: {ter_difference}\n_TER {model1}_: {first_score}\n_TER {model2}_: {second_score}\n" + \
            f"*Transcript*:\n{transcript}\n*ASR Output*:\n{asr_hypo}\n*Reference*:\n{ref_string}\n" + \
        f"*{model1} hypo*: {first_hypo}\n*{model2} hypo*: {second_hypo}"
        markdown_string += string
        markdown_string += "\n" + "---" * 15 + "\n"
    return markdown_string


def save_to_json(ter_content, filename):
    """Save data to a json file"""
    with open(f"{filename}.md", "w") as f:
        f.write(ter_content)



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
    result_filename = f"ter_comparison_{fairseq_file.split('.txt')[0]}_{fairseq_file_for_comparison.split('.txt')[0]}"
    references_list, hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(fairseq_file)
    asr_ref, asr_hypos = get_reference_and_hypothesis_strings_from_datafile(args.asr_output)
    ter_score_examples = calculate_ter_score_for_each_sentence_in_data(hypotheses_list, references_list, args.ast_data_set, asr_ref, asr_hypos)
    comparison_references_list, comparison_hypotheses_list = get_reference_and_hypothesis_strings_from_datafile(
        fairseq_file_for_comparison
    )
    comparison_ter_score_examples = calculate_ter_score_for_each_sentence_in_data(
        comparison_hypotheses_list,
        comparison_references_list,
        args.ast_data_set,
        asr_ref,
        asr_hypos
    )
    result_string = compare_ter_scores(
        ter_score_examples,
        comparison_ter_score_examples,
        args.model_one_name,
        args.model_two_name
    )
    save_to_json(result_string, result_filename)
