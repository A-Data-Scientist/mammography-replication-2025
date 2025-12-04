import os
os.environ["MPLBACKEND"] = "Agg" 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from utils import save_output_figure


def plot_confusion_matrix(cm: np.ndarray, fmt: str, label_encoder, is_normalised: bool, prefix="") -> None:
    """
    Plot confusion matrix.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param cm: Confusion matrix array.
    :param fmt: The formatter for numbers in confusion matrix.
    :param label_encoder: The label encoder used to get the number of classes.
    :param is_normalised: Boolean specifying whether the confusion matrix is normalised or not.
    :return: None.
    """
    title = str()
    if is_normalised:
        title = "Normalised Confusion Matrix {}".format(prefix)
        vmax = 1  # Y scale.
    elif not is_normalised:
        title = "Confusion Matrix {}".format(prefix)
        vmax = np.max(cm.sum(axis=1))  # Y scale.

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap=plt.cm.Blues, vmin=0, vmax=vmax)  # annot=True to annotate cells

    # Set labels, title, ticks and axis range.
    ax.set_xlabel('Predicted classes')
    ax.set_ylabel('True classes')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(label_encoder.classes_)
    ax.yaxis.set_ticklabels(label_encoder.classes_)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    bottom, top = ax.get_ylim()
    if is_normalised:
        save_output_figure("CM-normalised", prefix)
    elif not is_normalised:
        save_output_figure("CM", prefix)
    # plt.show()
    plt.close(fig)


def plot_comparison_chart(df: pd.DataFrame, prefix="") -> None:
    """
    Plot comparison bar chart.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param df: Compare data from json file.
    :return: None.
    """
    title = "Accuracy Comparison {}".format(prefix)

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='paper', y='accuracy', data=df)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    save_output_figure("Accuracy Comparison", prefix)
    # plt.show()
    plt.close(fig)


def plot_training_results(hist_input, plot_name: str, is_frozen_layers, prefix="") -> None:
    title = f"Training Loss on {config.dataset} {prefix}"
    if not is_frozen_layers:
        title += " (unfrozen layers)"

    plt.style.use("ggplot")

    # ---- Loss ----
    fig_loss, ax_loss = plt.subplots()
    n = len(hist_input.history["loss"])
    ax_loss.plot(np.arange(0, n), hist_input.history["loss"], label="train set")
    ax_loss.plot(np.arange(0, n), hist_input.history["val_loss"], label="validation set")
    ax_loss.set_title(title)
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Cross entropy loss")
    ax_loss.legend(loc="upper right")
    fig_loss.tight_layout()
    fig_loss.savefig(f"../output/dataset-{config.dataset}_model-{config.model}_{plot_name}-Loss.png")
    plt.close(fig_loss)

    # ---- Accuracy ----
    title_acc = f"Training Accuracy on {config.dataset}"
    if not is_frozen_layers:
        title_acc += " (unfrozen layers)"

    fig_acc, ax_acc = plt.subplots()
    if config.dataset == "mini-MIAS":
        ax_acc.plot(np.arange(0, n), hist_input.history["categorical_accuracy"], label="train set")
        ax_acc.plot(np.arange(0, n), hist_input.history["val_categorical_accuracy"], label="validation set")
    else:  # CBIS-DDSM or mini-MIAS-binary
        ax_acc.plot(np.arange(0, n), hist_input.history["binary_accuracy"], label="train set")
        ax_acc.plot(np.arange(0, n), hist_input.history["val_binary_accuracy"], label="validation set")

    ax_acc.set_title(title_acc)
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.1)
    ax_acc.legend(loc="upper right")
    fig_acc.tight_layout()
    fig_acc.savefig(f"../output/dataset-{config.dataset}_model-{config.model}_{plot_name}-Accuracy.png")
    plt.close(fig_acc)
