import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Experiment comparison selecting desired ones")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        default="output/Natural_Light-VIDEO_25mm-240605_124002",
        help="video name path",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        default=["ExpCalGray-OptGamma", "ExpNoCalGray-Poly2-Sep-Gamma"],
        help="experiment name list (space separated).",
    )
    args = parser.parse_args()

    video_name = Path(args.video)
    exp_folders = [Path(exp) for exp in args.experiments]

    plt.figure(figsize=(12, 6))
    sns.set_theme()

    # Plot experiment files
    for folder in exp_folders:
        filepath_txt = list(folder.glob("curve*.txt"))
        for filepath in filepath_txt:
            df = pd.read_csv(filepath, sep=";", decimal=".")
            sns.lineplot(x="Time(s)", y=" Reflectance ", data=df, label=folder.name)

    plt.xlabel("Time(s)")
    plt.ylabel("Reflectance(%)")
    plt.title("Reflectance vs Time(s)")
    plt.legend(title="Experiment")
    plt.savefig(video_name / "curve_comparison_Gamma_poly2.png")


if __name__ == "__main__":
    main()
