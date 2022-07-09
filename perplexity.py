from typing import List, Tuple
from math import exp


def get_log_probs_from_file(filename: str) -> List[str]:
    """get log probabilities from fairseq-generate log file"""
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    log_probs = []
    for line in data:
        if line.startswith("P-"):
            log_probs.append([float(prob) for prob in line.split()[1:]])
    return log_probs


def calculate_word_perplexity(log_prob_list):
    """Calculate perplexity of model for output"""
    total = sum([len(l) for l in log_prob_list])
    sentences = len(log_prob_list)
    sums = [sum(l_list) for l_list in log_prob_list]
    per_word_perplexity = 2 ** (
        -sum(sums) / total
    )  # fairseq-generate outputs with log_2
    per_sentence_perplexity = 2 ** (-sum(sums) / sentences)
    return per_word_perplexity, per_sentence_perplexity


l_p = get_log_probs_from_file(
    "score_references_covost_transcript_to_target_training_data.txt"
)
print(calculate_word_perplexity(l_p))
l_p = get_log_probs_from_file("score_references_covost_true_training_data.txt")
print(calculate_word_perplexity(l_p))

l_p = get_log_probs_from_file(
    "score_references_europarl_transcript_to_target_training_data.txt"
)
print(calculate_word_perplexity(l_p))
l_p = get_log_probs_from_file("score_references_europarl_true_training_data.txt")
print(calculate_word_perplexity(l_p))

l_p = get_log_probs_from_file("score_reference_mustc_training_data_original_data.txt")
print(calculate_word_perplexity(l_p))
