from pathlib import Path
from typing import List, Union

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Configure global matplotlib settings.
mpl.rc("axes", titlesize=36, labelsize=24, facecolor="none")
mpl.rc("xtick", labelsize=16)
mpl.rc("ytick", labelsize=16)
GIF_HEIGHT_PIXELS = 324
HEIGHT_PIXELS = 10 * GIF_HEIGHT_PIXELS
DPI = 600
HEIGHT = HEIGHT_PIXELS / DPI
WIDTH = 1.618 * HEIGHT
mpl.rc("figure", dpi=DPI, figsize=(WIDTH, HEIGHT), facecolor="none")
plt.style.use("ggplot")  # This plot style is borrowed from R's ggplot2.


def save_fig(
    outpath: Union[Path, str], tight_layout: bool = True, resolution: int = DPI
) -> None:
    """Save the current figure to a specified path.

    Args:
        outpath (Union[Path, str]): The desired output path for the image.
        tight_layout (bool, optional): Whether to crop the image to its content.
            Defaults to True.
        resolution (int, optional): The pixel density of the image. Defaults to `DPI`.
    """
    if tight_layout:
        plt.tight_layout()
    plt.savefig(outpath, dpi=resolution, transparent=True)
    print(f"Figure saved to: {outpath}")


def prepare_data_for_plotting(history: List[int], window_len: int = 2) -> pd.DataFrame:
    """Put historical total rewards data into a dataframe.

    Args:
        history (List[int]): A sequential list of high scores for each game played.
        window_len (int, optional): The window over which to calculate rolling
            statistics. Defaults to 5.

    Returns:
        pd.DataFrame: Dataframe containing a list of total reward scores for each game
            played.
    """
    df = pd.DataFrame(history, columns=["Total Reward"])
    df["Total Reward Rolling Trend"] = df["Total Reward"].rolling(window_len).mean()
    df["Episode"] = range(len(history))
    return df


def plot_history(
    history: List[int], params: dict, outpath: Union[Path, str] = "learning-curve.png"
) -> None:
    """Plot total rewards history for all games played.

    Args:
        history (List[int]): A list of total reward scores achieved by the agent
            corresponding to each game it played.
        params (dict): Dictionary of fitting parameters used for all training epochs.
        outpath (Union[Path, str], optional): The desired output path for the generated
            figure. Defaults to "learning-curve.png".
    """
    window_len = max(int(params.get("num_episodes") / 5), 5)
    df = prepare_data_for_plotting(history, window_len=window_len)
    _, ax = plt.subplots()
    # Establish the base scatter plot.
    ax.plot(
        df["Episode"],
        df["Total Reward"],
        ".",
        label="Total Reward",
        color="royalblue",
    )
    ax.plot(
        df["Episode"],
        df["Total Reward Rolling Trend"],
        "--",
        label="Total Reward Rolling Trend",
        color="grey",
    )
    # Perform the linear regression and plot the line of best fit.
    X = df["Episode"].values.reshape(-1, 1)
    y = df["Total Reward"].dropna().values
    X_filtered = X[: len(y)]  # Ensure `X` and `y` have the same length.
    model = LinearRegression().fit(X_filtered, y)
    trendline = model.predict(X_filtered)
    ax.plot(
        X_filtered,
        trendline,
        "-",
        alpha=0.5,
        label="Total Reward Linear Trend",
        color="skyblue",
    )
    ax.set_title("Snake Agent Learning Curve")
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Total Episode Reward")
    plt.legend(loc="best")
    outpath = outpath if outpath is not None else "learning-curve.png"
    save_fig(outpath)
    plt.close("all")
