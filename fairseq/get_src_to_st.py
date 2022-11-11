#!/usr/bin/env python3 

import pandas as pd
import numpy as np



def get_src_in_frame_on_join(tgt_tsv, src_tsv):
    src_df = pd.read_csv(src_tsv, sep="\t")
    tgt_df = pd.read_csv(tgt_tsv, sep="\t")
    src_df.id = src_df.id.str.strip()
    tgt_df.id = tgt_df.id.str.strip()
    tgt_df.audio = tgt_df.audio.str.strip()
    src_df.audio = src_df.audio.str.strip()
    tgt_df.speaker = tgt_df.speaker.str.strip()
    src_df.speaker = src_df.speaker.str.strip()
    tgt_df = tgt_df[tgt_df['id'].isin(src_df['id'])]
    src_df = src_df[src_df['id'].isin(tgt_df['id'])]
    df = pd.merge(tgt_df, src_df, on=["id", "audio", "n_frames", "speaker"], how='left')
    # new_df = pd.merge(tgt_df, src_df, on=["id", "audio", "speaker", "n_frames"], how="inner")
    # new_df = tgt_df.set_index("id").join(src_df.set_index("id"), on="id", lsuffix='_caller')
    df.columns = ['src_text' if x=='tgt_text_y' else x for x in df.columns]
    df.columns = ['tgt_text' if x=='tgt_text_x' else x for x in df.columns]
    df.to_csv(f"{tgt_tsv.split('.tsv')[0]}_with_source_text.tsv", sep="\t", index=False)


def get_src_in_frame(tgt_tsv, src_tsv):
    src_df = pd.read_csv(src_tsv, sep="\t")
    tgt_df = pd.read_csv(tgt_tsv, sep="\t")
    id_list = tgt_df.id.tolist()
    src_df = src_df[src_df['id'].isin(tgt_df['id'])]
    tgt_df = tgt_df[tgt_df['id'].isin(src_df['id'])]

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
    dataframe["src_text"].replace('', np.nan)
    dataframe = dataframe.dropna()
    dataframe.to_csv(f"{tgt_tsv.split('.tsv')[0]}_with_source_text.tsv", sep="\t", index=False)


get_src_in_frame("/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/train_st.tsv", "/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/train_asr.tsv")
get_src_in_frame_on_join("/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/dev_st.tsv", "/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/dev_asr.tsv")
get_src_in_frame_on_join("/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/tst-COMMON_st.tsv", "/home/rebekka/t2b/Projekte/ma/test_preprocessing/imitkd_ast/fairseq/MUSTC_data/en-de/tst-COMMON_asr.tsv")
