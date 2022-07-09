import pandas as pd

from pathlib import Path
from data_utils import gen_vocab

from pathlib import Path


def get_column_text(filename, column_name="tgt_txt"):
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
    if column_name == "tgt_txt":
        index = 4
        if column_number != 6:
            index = 3
    elif column_name == "src_txt":
        index = -1
    else:
        raise ValueError(f"{column_name} is not supported")
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    tgt_text = [line.split("\t")[index].strip() for line in data[1:]]
    return "\n".join(tgt_text)


def add_tokenized_text_to_target_old_fashioned(filename, tokenized_text, result_filename, column_name="tgt_txt"):
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    if column_name == "tgt_txt":
        new_csv = [data[0]]
    elif column_name == "src_txt":
        new_csv = [data[0]  + "\t" + "src_text"]
    else:
        raise ValueError(f"{column_name} is not supported")
    tokenized_text = tokenized_text.split("\n")[:-1]
    for i, line in enumerate(data[1:]):
        line = line.split("\t")
        if column_name == "tgt_txt":
            new_csv.append("\t".join(line[:3]).strip() + "\t" + tokenized_text[i] + "\t" + "\t".join(line[4:]).strip())
        else:
            new_csv.append("\t".join(line[:5]).strip() + "\t" + tokenized_text[i])
    with open(result_filename, "w") as f:
        f.write("\n".join(new_csv))


def add_tokenized_text_to_target_old_fashioned_train(filename, tokenized_text, result_filename, column_name="tgt_txt"):
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    if column_name == "tgt_txt":
        new_csv = ["\t".join(data[0].split("\t")[:3] + data[0].split("\t")[4:])]
    elif column_name == "src_txt":
        new_csv = [data[0]]
    else:
        raise ValueError(f"{column_name} is not supported")
    tokenized_text = tokenized_text.split("\n")[:-1]
    for i, line in enumerate(data[1:]):
        line = line.split("\t")
        if column_name == "tgt_txt":
            new_csv.append("\t".join(line[:3]).strip() + "\t" + tokenized_text[i] + "\t" + "\t".join(line[5:]).strip())
        else:
            new_csv.append("\t".join(line[:5]).strip() + "\t" + tokenized_text[i].strip())
    with open(result_filename, "w") as f:
        f.write("\n".join(new_csv))


def get_text_from_datasets():
    target_text = get_column_text("covost/en/train_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/train_text.txt", target_text)
    target_text = get_column_text("covost/en/dev_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/dev_text.txt", target_text)
    target_text = get_column_text("covost/en/test_st_en_de_with_source_text.tsv")
    save_text_in_file("covost/test_text.txt", target_text)
    target_text = get_column_text("covost/en/dev_st_en_de_with_source_text.tsv", column_name="src_txt")
    save_text_in_file("covost/dev_text_en.txt", target_text)
    target_text = get_column_text("covost/en/test_st_en_de_with_source_text.tsv", column_name="src_txt")
    save_text_in_file("covost/test_text_en.txt", target_text)
    target_text = get_source_text_old_fashioned("mustc_csvs/train_st_with_source_text.tsv")
    save_text_in_file("mustc_processed/train_text.txt", target_text)
    target_text = get_source_text_old_fashioned("mustc_csvs/dev_st_with_source_text.tsv", column_number=5)
    save_text_in_file("mustc_processed/dev_text.txt", target_text)
    target_text = get_column_text("mustc_csvs/tst-COMMON_st_with_source_text.tsv")
    save_text_in_file("mustc_processed/test_text.txt", target_text)
    target_text = get_source_text_old_fashioned("mustc_csvs/train_st_with_source_text_normal.tsv", column_name="src_txt")
    save_text_in_file("mustc_processed/train_text_en.txt", target_text)
    target_text = get_column_text("mustc_processed/dev_processed.tsv", column_name="src_txt")
    save_text_in_file("mustc_processed/dev_text_en.txt", target_text)
    target_text = get_column_text("mustc_processed/tst-COMMON_processed.tsv", column_name="src_txt")
    save_text_in_file("mustc_processed/test_text_en.txt", target_text)
    source_text = get_column_text("covost/en/train_st_en_de_with_source_text_normal.tsv", column_name="src_txt")
    save_text_in_file("covost/train_text_en.txt", source_text)


def create_processed_datasets():
    
    gen_vocab(Path("covost_processed_text/train.tok.de"), Path("covost/en/spm_bpe8000_ast"), "bpe", 8000)

    tokenized_training_text = get_tokenized_text("covost_processed_text/train.tok.de")
    add_tokenized_text_to_target('covost/en/train_st_en_de_with_source_text.tsv', tokenized_training_text,
                                 'covost/en/train_processed.tsv')
    tokenized_training_text = get_tokenized_text("covost_processed_text/train_asr.tok.en")
    add_tokenized_text_to_target("covost/en/train_asr_en.tsv", tokenized_training_text,
                                 "covost/en/train_asr_processed.tsv", column_name="tgt_txt")
    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev.tok.de")
    add_tokenized_text_to_target('covost/en/dev_st_en_de_with_source_text.tsv', tokenized_dev_text,
                                 'covost/en/dev_processed.tsv')
    tokenized_dev_text = get_tokenized_text("covost_processed_text/dev_asr.tok.en")
    add_tokenized_text_to_target('covost/en/dev_asr_en.tsv', tokenized_dev_text,
                                 'covost/en/dev_asr_processed.tsv', column_name="tgt_txt")
    tokenized_test_text = get_tokenized_text('covost_processed_text/test.tok.de')
    add_tokenized_text_to_target('covost/en/test_st_en_de_with_source_text.tsv', tokenized_test_text,
                                 'covost/en/test_processed.tsv')
    tokenized_test_text = get_tokenized_text('covost_processed_text/test_asr.tok.en')
    add_tokenized_text_to_target('covost/en/test_asr_en.tsv', tokenized_test_text,
                                 'covost/en/test_asr_processed.tsv', column_name="tgt_txt")
   
    gen_vocab(Path("mustc_processed_text/train.tok.de"), Path("MUST/en-de/spm_bpe8000_ast"), "bpe", 8000)
    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.de")
    add_tokenized_text_to_target_old_fashioned_train("mustc_csvs/train_st_with_source_text.tsv", tokenized_training_text,
                                                     "mustc_processed/train_processed.tsv")
    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.de")
    add_tokenized_text_to_target_old_fashioned("mustc_csvs/dev_st_with_source_text.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_processed.tsv")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.de")
    add_tokenized_text_to_target_old_fashioned("mustc_csvs/tst-COMMON_st_with_source_text.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_processed.tsv")
    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned_train("mustc_csvs/train_asr.tsv", tokenized_training_text,
                                                     "mustc_processed/train_asr_processed.tsv")
    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_csvs/dev_asr.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_asr_processed.tsv")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_csvs/tst-COMMON_asr.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_asr_processed.tsv")

    tokenized_training_text = get_tokenized_text("mustc_processed_text/train.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_processed/train_processed.tsv", tokenized_training_text,
                                               "mustc_processed/train_processed.tsv", column_name="src_txt")

    tokenized_dev_text = get_tokenized_text("mustc_processed_text/dev.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_processed/dev_processed.tsv", tokenized_dev_text,
                                               "mustc_processed/dev_processed.tsv", column_name="src_txt")
    tokenized_test_text = get_tokenized_text("mustc_processed_text/test.tok.en")
    add_tokenized_text_to_target_old_fashioned("mustc_processed/tst-COMMON_processed.tsv", tokenized_test_text,
                                               "mustc_processed/tst-COMMON_processed.tsv", column_name="src_txt")
    tokenized_training_text = get_tokenized_text("../../../v1.1/en/de/train/segments.tok.en")
    add_tokenized_text_to_target_old_fashioned("../../../v1.1/train_st_en_de.tsv", tokenized_training_text,
                                               "../../../v1.1/train_processed.tsv", column_name="src_txt")
    tokenized_train_noisy_text = get_tokenized_text("../../../v1.1/en/de/train-noisy/segments.tok.en")
    add_tokenized_text_to_target_old_fashioned("../../../v1.1/train-noisy_st_en_de.tsv", tokenized_train_noisy_text,
                                               "../../../v1.1/train-noisy_processed.tsv", column_name="src_txt")
    tokenized_test_text = get_tokenized_text("../../../v1.1/en/de/dev/segments.tok.en")
    add_tokenized_text_to_target_old_fashioned("../../../v1.1/dev_st_en_de.tsv", tokenized_test_text,
                                               "../../../v1.1/dev_processed.tsv", column_name="src_txt")
    tokenized_test_text = get_tokenized_text("../../../v1.1/en/de/test/segments.tok.en")
    add_tokenized_text_to_target_old_fashioned("../../../v1.1/test_st_en_de.tsv", tokenized_test_text,
                                               "../../../v1.1/test_processed.tsv", column_name="src_txt")


if __name__ == "__main__":
    if not Path("covost_processed_text/train.tok.de").is_file():
        get_text_from_datasets()
        raise OSError("pre-processing with moses-tokenizer and perl script is required - run prepare-rest.sh before "
                      "continuing")
    create_processed_datasets()
