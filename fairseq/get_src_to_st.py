#!/usr/bin/env python3 

import pandas as pd


def get_src_in_frame(tgt_tsv, src_tsv):
    src_df = pd.read_csv(src_tsv, sep="\t")
    tgt_df = pd.read_csv(tgt_tsv, sep="\t")
    id_list = tgt_df.id.tolist()
    src_list = src_df.values.tolist()
    tgt_list = tgt_df.values.tolist()
    source_in_order = ["" for _ in tgt_list]
    for list_item in src_list:
        ids = list_item[0]
        try:
            tgt_index = id_list.index(ids)
            source_in_order[tgt_index] = list_item[3]
        except ValueError:
            print(list_item)
            continue
    dataframe = tgt_df.assign(src_text=pd.Series(source_in_order))
    dataframe.to_csv(f"{tgt_tsv.split('.tsv')[0]}_with_source_text.tsv", sep="\t")


get_src_in_frame("/scratch/hubert/data/covost/en/test_st_en_de.tsv", "/scratch/hubert/data/covost/en/test_asr_en.tsv")
get_src_in_frame("/scratch/hubert/data/covost/en/dev_st_en_de.tsv", "/scratch/hubert/data/covost/en/dev_asr_en.tsv")
get_src_in_frame("/scratch/hubert/data/covost/en/train_st_en_de.tsv", "/scratch/hubert/data/covost/en/train_asr_en.tsv")
get_src_in_frame("/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/train_st.tsv", "/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/train_asr.tsv")
get_src_in_frame("/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/dev_st.tsv", "/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/dev_asr.tsv")
get_src_in_frame("/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/tst-COMMON_st.tsv", "/scratch/hubert/fairs/fairseq/examples/speech_to_text/MUST/en-de/tst-COMMON_asr.tsv")
