#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import probscale
import matplotlib.gridspec as gridspec
from fairseq.data.dictionary import Dictionary


def get_data(
    filename="fairseq/wmt19.en-de.joined-dict.ensemble/dict.en.txt",
    csv_filename="covost_wer_0_5.csv",
):
    """get vocabulary into dict of fairseq dictionary and read log frame"""
    index2word = Dictionary.load(filename)
    dataframe = pd.read_csv(csv_filename, sep="\t")
    return index2word, dataframe


def create_probs_plot(
    dataframe_row, np_array, index2word, image_folder, show_transcript=True
):
    """create standard plot of top 8 probabilities for each time step for both original transcript and genareted"""
    for index, row in enumerate(np_array):
        if np.max(row) == 0:
            last_index = index
            break
        last_index = index
    np_array = np_array[:last_index, :]
    topk_indices = np.argsort(np_array, axis=1)[:, -8:]
    nrows = np_array.shape[0] / 4
    if int(nrows) < nrows:
        nrows = 1 + int(nrows)
    else:
        nrows = int(nrows)
    fig = plt.figure(figsize=(16, 12), constrained_layout=True, tight_layout=True)
    gs = gridspec.GridSpec(nrows, 4, hspace=0.9)
    transcript = dataframe_row.transcript
    target = dataframe_row.target
    prefix_at_timestep = target.split()
    correct_transcript = dataframe_row["original source text"]

    for row_number, prob_at_timestep in enumerate(np_array):
        x_labels = [index2word[p] for p in topk_indices[row_number]]
        top_k_probs = [prob_at_timestep[topk] for topk in topk_indices[row_number]]
        ax = plt.subplot(gs[row_number])
        if row_number <= 5:
            ax.set_title(
                f"prefix: {' '.join(prefix_at_timestep[:row_number])}", {"fontsize": 10}
            )
        else:
            ax.set_title(
                f"prefix: [...]{' '.join(prefix_at_timestep[row_number-5:row_number])}",
                {"fontsize": 10},
            )
        ax.bar([i for i in range(8)], top_k_probs, max(top_k_probs))
        ax.set_xticks([i for i in range(8)], x_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if show_transcript:
        fig.suptitle(
            f"correct transcript: {correct_transcript}\ntranscript: {transcript}\ntarget: {target}"
        )
    else:
        fig.suptitle(f"correct transcript: {correct_transcript}\ntarget: {target}")
    plt.legend()
    fig.supylabel("Probability", fontsize=25)
    fig.supxlabel("Output BP", fontsize=25)
    # plt.show()
    plt.savefig(
        f"{image_folder}/{transcript}_{correct_transcript}_tok_k_probs.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def choose_row_and_numpy_file(
    dataframe, np_folder, index2word, wer, save_folder, to_keep=15
):
    """Create histogram of WER (subset above 0) and plot per-step expert probabilities of to_keep samples"""
    old_counter = None
    relevant_data_rows = dataframe.loc[dataframe.WER > wer]
    relevant_data_rows = relevant_data_rows.head(100)
    print(len(relevant_data_rows))
    plt.rc("font", size=45)
    plt.rc("axes", labelsize=30)
    plt.hist(
        relevant_data_rows.WER.clip(upper=100).values,
        bins="auto",
        rwidth=0.9,
        color="#0504aa",
    )
    plt.grid()
    plt.xlabel("WER", fontsize=65)
    plt.ylabel("Frequency", fontsize=65)
    plt.show()
    relevant_data_rows = relevant_data_rows.loc[dataframe.WER > 0]
    print(len(relevant_data_rows))
    sorted_relevant_data_rows = relevant_data_rows.sort_values("WER", ascending=False)[
        :to_keep
    ]
    print(sorted_relevant_data_rows)
    for _, data_row in sorted_relevant_data_rows.iterrows():
        file_number = data_row.counter
        sample_index = data_row["in data counter"]
        if file_number != old_counter:
            np_array = np.load(f"{np_folder}/{file_number}_covost_from_transcripts.npy")
            np_array_orig = np.load(f"{np_folder}/{file_number}_covost_original.npy")
            old_counter = file_number
        create_probs_plot(
            data_row, np.squeeze(np_array[[sample_index]], 0), index2word, save_folder
        )
        create_probs_plot(
            data_row,
            np.squeeze(np_array_orig[[sample_index]], 0),
            index2word,
            save_folder + "_the_original",
            show_transcript=False,
        )


# load vocabulary into fairseq dictionary

dictionary, data = get_data(csv_filename="covost_wers.csv")

# plot WER-histogram of entire dataset
plt.rc("font", size=45)
plt.rc("axes", labelsize=30)
plt.hist(data.WER.clip(upper=100).values, bins="auto", rwidth=0.9, color="#0504aa")
plt.grid()
plt.xlabel("WER")
plt.ylabel("Frequency")
plt.show()
plt.hist(
    data.WER.clip(upper=100).values,
    bins="auto",
    rwidth=0.9,
    color="#0504aa",
    density=True,
)
plt.grid()
plt.xlabel("WER")
plt.ylabel("Density")
plt.rc("font", size=15)
plt.rc("axes", labelsize=20)
plt.show()
wer = 90
# plot probabilities
dictionary, data = get_data(csv_filename="covost_wer_all.csv")
choose_row_and_numpy_file(
    data,
    "expert_probs_kd_on_translations_few",
    dictionary,
    0,
    "asrkd_sample_100_above_0",
    to_keep=100,
)