from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_analisys(mcontrast, mcontrast_corregido, vopath):
    plt.figure(figsize=(10, 5))
    sns.set_theme()
    sns.lineplot(x=range(len(mcontrast)), y=mcontrast, label="Conv Std")
    sns.lineplot(x=range(len(mcontrast_corregido)), y=mcontrast_corregido, label="Conv Std Satured")
    plt.xlabel("Frame")
    plt.ylabel("Values")
    plt.title("Analysis Graphs")
    plt.legend()
    # Save
    plt.savefig(Path(vopath) / "analysis.jpg")
    plt.close()


def plot_channel(plot_color, optgamma, vopath):
    ori_m, ori_g, color = plot_color
    if color == "Y":
        color = "red"

    # Plot red comparison:
    plt.figure(figsize=(10, 5))
    plt.plot(ori_m, label=f"Original {color}", color=color, linestyle="-")
    plt.plot(ori_g, label=f"Modified {color} (Gamma)", color="orange", linestyle="-")
    plt.title(f"{color} vs {color} after {optgamma} Gamma correction")
    plt.xlabel("Frame")
    plt.ylabel(f"{color}")
    plt.legend()
    # Save red values plot
    plt.savefig(Path(vopath) / f"{color}_channel_comparison.jpg")


def save_channel(plot_color, vopath):
    ori_m, ori_g, color = plot_color
    # Save  channel values in txt file
    with open(Path(vopath) / f"{color}_channel_values.txt", "w") as f:
        f.write(f"Original{color}\tGamma{color}\n")
        for original, gamma_corrected in zip(ori_m, ori_g, strict=False):
            f.write(f"{original}\t{gamma_corrected}\n")
