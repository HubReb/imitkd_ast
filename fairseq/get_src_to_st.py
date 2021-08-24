#!/usr/bin/env python3 

import pandas as pd


def get_src_in_frame(src_tsv, tgt_tsv):
    src_df = pd.read_csv(src_tsv, sep="\t")
    id_list = src_df.id.tolist()
    tgt_list = pd.read_csv(tgt_tsv, sep="\t").values.tolist()
    translations_in_order = ["" for _ in tgt_list]
    for list_item in tgt_list:
        ids = list_item[0]
        translations_in_order[id_list.index(ids)] = list_item[3]
    dataframe = src_df.assign(src_text=pd.Series(translations_in_order))
    dataframe.to_csv(f"{src_tsv.split('.tsv')[0]}_with_source_text.tsv", sep="\t")


get_src_in_frame("test_st_en_de.tsv", "test_asr_en.tsv")
get_src_in_frame("dev_st_en_de.tsv", "dev_asr_en.tsv")
get_src_in_frame("train_st_en_de.tsv", "train_asr_en.tsv")
