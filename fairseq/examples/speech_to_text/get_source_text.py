import pandas as pd

from pathlib import Path
from data_utils import gen_vocab


def get_source_text(filename):
    df = pd.read_csv(filename, sep="\t")
    text = "\n".join(df.tgt_text.tolist())
    return text


def save_text_in_file(filename, text):
    with open(filename, "w") as f:
        f.write(text)


def get_tokenized_text(filename):
    with open(filename) as f:
        data = f.read()
    return data


def add_tokenized_text_to_target(filename, tokenized_text, result_filename):
    df = pd.read_csv(filename, sep="\t")
    df.tgt_text = tokenized_text.split("\n")[:-1]
    try:
        df = df.drop(columns=["pr_tgt_text"])
    except KeyError:
        # we never put wmt19 target text in here
        pass
    df.to_csv(result_filename, sep="\t")


if __name__ == "__main__":
    target_text = get_source_text("covost/en/train_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/train_text.txt", target_text)
    target_text = get_source_text("covost/en/dev_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/dev_text.txt", target_text)
    target_text = get_source_text("covost/en/test_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/test_text.txt", target_text)
    gen_vocab(Path("covost_processed_text/train.tok.de"), Path("covost/en/spm_bpe8000_ast"), "bpe", 8000)

    tokenized_training_text = get_tokenized_text("covost_processed_text/train.tok.de")
    add_tokenized_text_to_target('covost/en/train_st_en_de_with_source_text.tsv', tokenized_training_text,
                                 'covost/en/train_processed.tsv')
    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev.tok.de")
    add_tokenized_text_to_target('covost/en/dev_st_en_de_with_source_text.tsv', tokenized_dev_text,
                                 'covost/en/dev_processed.tsv')
    tokenized_test_text = get_tokenized_text('covost_processed_text/test.tok.de')
    add_tokenized_text_to_target('covost/en/test_st_en_de_with_source_text.tsv', tokenized_test_text,
                                 'covost/en/test_processed.tsv')
