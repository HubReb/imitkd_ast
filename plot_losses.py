#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_data_and_filter_logged_data(logging_file):
    """extract validation loss and validation steps from fairseq-train log file"""
    with open(logging_file) as f:
        data = f.read()
    dev_loss_lines = [
        line
        for line in data.split("\n")
        if line.startswith("epoch") and "dev_processed" in line
    ]
    steps, losses = {}, []
    last_epoch = ""
    best_loss = 999
    best_step = 0
    i = 0
    for logged_line in dev_loss_lines:
        epoch = int(logged_line.split("|")[0].split("epoch ")[1].strip())
        if epoch == last_epoch:
            continue
        last_epoch = epoch
        split_logged_line = logged_line.split("|")
        if "best_loss" in split_logged_line[-1]:
            val_loss = float(split_logged_line[-1].split()[1])
            step = split_logged_line[-2].split()[1]
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
        else:
            step = split_logged_line[-1].split()[1]
        loss = float(logged_line.split("|")[2].split()[1])
        steps[i] = step
        i += 1
        losses.append(loss)
    return steps, losses


def draw_graph(steps, loss, asr_steps, asr_losses):
    """draw validation losses on graph and set x-ticks dynamically to number of data points"""
    fig, ax = plt.subplots(constrained_layout=True)
    xs = range(len(loss))
    assert steps == asr_steps

    # adapted from https://matplotlib.org/2.2.2/gallery/ticks_and_spines/tick_labels_from_values.html
    def format_fn(
        tick_val, tick_pos
    ):  # need to set second argument to pass to set_major_formatter
        if int(tick_val) in xs:
            return steps[int(tick_val)]
        else:
            return ""

    ax.xaxis.set_major_formatter(format_fn)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(loss, label="ImitKD")
    ax.plot(asr_losses, label="ImitKDT")
    ax.grid()
    ax.set_xlabel("number of training steps")
    ax.set_ylabel("development loss")
    # ax.set_title("Validation loss")
    ax.legend()
    plt.show()
    # plt.savefig("imit_losses.png")


training_steps, val_losses = load_data_and_filter_logged_data(
    "europarl_asrimitkd_training_process.txt"
)
asr_training_steps, asr_val_losses = load_data_and_filter_logged_data(
    "europarl_imitkd_training_process.txt"
)
draw_graph(training_steps, val_losses, asr_training_steps, asr_val_losses)
