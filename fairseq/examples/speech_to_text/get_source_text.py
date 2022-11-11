import pandas as pd

from pathlib import Path
from data_utils import gen_vocab

from pathlib import Path


def get_column_text(filename, column_name="tgt_txt"):
    """get text data in one column"""
    df = pd.read_csv(filename, sep="\t")
    if column_name == "tgt_txt":
        text = "\n".join(df.tgt_text.tolist())
    elif column_name == "src_txt":
        text = "\n".join(df.src_text.tolist())
    else:
        raise ValueError(f"{column_name} is not supported")
    return text


def save_text_in_file(filename, text):
    with open(filename, "w") as f:
        f.write(text)


def get_tokenized_text(filename):
    with open(filename) as f:
        data = f.read()
    return data


def add_tokenized_text_to_target(filename, tokenized_text, result_filename, column_name="tgt_txt"):
    """Replace non-preprocessed text with pre-processed text for either target or source"""
    df = pd.read_csv(filename, sep="\t")
    if column_name == "tgt_txt":
        print(len(df.tgt_text),  len(tokenized_text.split("\n")[:-1]))
        df.tgt_text = tokenized_text.split("\n")[:-1]
    elif column_name == "src_txt":
        df.src_text = tokenized_text.split("\n")[:-1]
    else:
        raise ValueError(f"{column_name} is not supported")
    try:
        df = df.drop(columns=["pr_tgt_text"])
    except KeyError:
        # we never put wmt19 target text in here
        pass
    df.to_csv(result_filename, sep="\t")


def get_source_text_old_fashioned(filename, column_name="tgt_txt", column_number=6):
    """ to be used if pandas refuses to recognizes seperators"""
    if column_name == "tgt_txt":
        index = 3
    elif column_name == "src_txt":
        index = -2
    else:
        raise ValueError(f"{column_name} is not supported")
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    tgt_text = [line.split("\t")[index].strip() for line in data[1:]]
    return "\n".join(tgt_text)


def add_tokenized_text_to_target_old_fashioned(filename, tokenized_text, result_filename, column_name="tgt_txt"):
    """ Replace non-preprocessed text with pre-processed text for either target or source: To be used if csv cannot be loaded with pandas"""
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    new_csv = [data[0]]
    tokenized_text = tokenized_text.split("\n")[:-1]
    for i, line in enumerate(data[1:]):
        line = line.split("\t")
        if column_name == "tgt_txt":
            new_csv.append("\t".join(line[:3]).strip() + "\t" + tokenized_text[i] + "\t" + "\t".join(line[4:]).strip())
        else:
            new_csv.append("\t".join(line[:4]).strip() + "\t" + tokenized_text[i] + "\t" +  "\t".join(line[5:]).strip())
    with open(result_filename, "w") as f:
        f.write("\n".join(new_csv))


def get_text_from_datasets(mustc_folder, covost_folder):
    """ Extract src and target text for all files """
    if not Path("mustc_processed/").exists():
        import os
        os.makedirs("mustc_processed/")
    target_text = get_source_text_old_fashioned(f"{mustc_folder}/train_st_with_source_text.tsv")
    save_text_in_file("mustc_processed/train_text.txt", target_text)
    target_text = get_column_text(f"{mustc_folder}/dev_st_with_source_text.tsv")
    save_text_in_file(f"mustc_processed/dev_text.txt", target_text)
    target_text = get_column_text(f"{mustc_folder}/tst-COMMON_st_with_source_text.tsv")
    save_text_in_file(f"mustc_processed/test_text.txt", target_text)
    target_text = get_source_text_old_fashioned(f"{mustc_folder}/train_st_with_source_text.tsv", column_name="src_txt")
    save_text_in_file(f"mustc_processed/train_text_en.txt", target_text)
    target_text = get_column_text(f"{mustc_folder}/dev_st_with_source_text.tsv", column_name="src_txt")
    save_text_in_file(f"mustc_processed/dev_text_en.txt", target_text)
    target_text = get_column_text(f"{mustc_folder}/tst-COMMON_st_with_source_text.tsv", column_name="src_txt")
    save_text_in_file(f"mustc_processed/test_text_en.txt", target_text)
    if not Path("covost_processed/").exists():
        import os
        os.makedirs("covost_processed/")
    target_text = get_column_text(f"{covost_folder}/train_st_en_de_with_source_text.tsv")
    save_text_in_file("covost_processed/train_text.txt", target_text)
    target_text = get_column_text(f"{covost_folder}/dev_st_en_de_with_source_text.tsv")
    save_text_in_file("covost_processed/dev_text.txt", target_text)
    target_text = get_column_text(f"{covost_folder}/test_st_en_de_with_source_text.tsv")
    save_text_in_file("covost_processed/test_text.txt", target_text)
    target_text = get_column_text(f"{covost_folder}/train_st_en_de_with_source_text.tsv", column_name="src_txt")
    save_text_in_file("covost_processed/train_text_en.txt", target_text)
    target_text = get_column_text(f"{covost_folder}/dev_st_en_de_with_source_text.tsv", column_name="src_txt")
    save_text_in_file("covost_processed/dev_text_en.txt", target_text)
    target_text = get_column_text(f"{covost_folder}/test_st_en_de_with_source_text.tsv", column_name="src_txt")
    save_text_in_file("covost_processed/test_text_en.txt", target_text)

def create_processed_datasets(mustc_folder, covost_folder):
    """ Replace source and target text with pre-processed source and target for all files """
    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.de")
    add_tokenized_text_to_target_old_fashioned(f"{mustc_folder}/train_st_with_source_text.tsv", tokenized_training_text,
                                                     "mustc_processed/train_processed.tsv")
    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.de")
    add_tokenized_text_to_target(f"{mustc_folder}/dev_st_with_source_text.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_processed.tsv")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.de")
    add_tokenized_text_to_target(f"{mustc_folder}/tst-COMMON_st_with_source_text.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_processed.tsv")
    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned(f"{mustc_folder}/train_asr.tsv", tokenized_training_text,
                                                     "mustc_processed/train_asr_processed.tsv")
    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_processed/train_processed.tsv", tokenized_training_text,
                                               "mustc_processed/train_processed.tsv", column_name="src_txt")

    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.en")
    add_tokenized_text_to_target("mustc_processed/dev_processed.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_processed.tsv", column_name="src_txt")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.en")
    add_tokenized_text_to_target("mustc_processed/tst-COMMON_processed.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_processed.tsv", column_name="src_txt")
    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.en")
    add_tokenized_text_to_target_old_fashioned(f"{mustc_folder}/dev_asr.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_asr_processed.tsv")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.en")
    add_tokenized_text_to_target(f"{mustc_folder}/tst-COMMON_asr.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_asr_processed.tsv")


    tokenized_training_text = get_tokenized_text("covost_processed_text/train.tok.de")
    add_tokenized_text_to_target_old_fashioned(f"{covost_folder}/train_st_en_de_with_source_text.tsv", tokenized_training_text,
                                                     "covost_processed/train_processed.tsv")
    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev.tok.de")
    add_tokenized_text_to_target(f"{covost_folder}/dev_st_en_de_with_source_text.tsv", tokenized_dev_text,
                                               "covost_processed/dev_processed.tsv")
    tokenized_test_text = get_tokenized_text("covost_processed_text/test.tok.de")
    add_tokenized_text_to_target(f"{covost_folder}/test_st_en_de_with_source_text.tsv", tokenized_test_text,
                                               "covost_processed/test_processed.tsv")
    tokenized_training_text = get_tokenized_text("covost_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned(f"{covost_folder}/train_asr_en.tsv", tokenized_training_text,
                                                     "covost_processed/train_asr_processed.tsv")
    tokenized_training_text = get_tokenized_text("covost_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned("covost_processed/train_processed.tsv", tokenized_training_text,
                                               "covost_processed/train_processed.tsv", column_name="src_txt")

    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev.tok.en")
    add_tokenized_text_to_target("covost_processed/dev_processed.tsv", tokenized_dev_text,
                                               "covost_processed/dev_processed.tsv", column_name="src_txt")
    tokenized_test_text = get_tokenized_text("covost_processed_text/test.tok.en")
    add_tokenized_text_to_target("covost_processed/test_processed.tsv", tokenized_test_text,
                                               "covost_processed/test_processed.tsv", column_name="src_txt")
    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev.tok.en")
    add_tokenized_text_to_target_old_fashioned(f"{covost_folder}/dev_asr_en.tsv", tokenized_dev_text,
                                               "covost_processed/dev_asr_processed.tsv")
    tokenized_test_text = get_tokenized_text("covost_processed_text/test.tok.en")
    add_tokenized_text_to_target(f"{covost_folder}/test_asr_en.tsv", tokenized_test_text,
                                               "covost_processed/test_asr_processed.tsv")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mustc-data", "-m", required=True, type=str,
        help="Folder containing the MuST-C speech-to-text datasets (e.g. train_st_with_source_text)"
    )
    parser.add_argument(
        "--covost-data", "-c", required=True, type=str,
        help="Folder containing the CoVoST 2 speech-to-text datasets (e.g. train_st_with_source_text)"
    )
    args = parser.parse_args()
    if not Path("mustc_processed_text/train.tok.de").is_file():
        get_text_from_datasets(args.mustc_data, args.covost_data)
        raise OSError("pre-processing with moses-tokenizer and perl script is required - run prepare-rest.sh before "
                       "continuing")
    create_processed_datasets(args.mustc_data, args.covost_data)
