from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from gif_builder import GifBuilder

RUNS_DIR = Path("runs")


def make_gif_from_images(instance_folder: Union[Path, str] = None):
    """Loop through eps files and turn them into an animated gif.

    Args:
        instance_folder (Union[Path, str]): Folder containing all data for a single run.
    """
    if instance_folder is None:
        most_recent_dir = max(RUNS_DIR.iterdir(), key=lambda d: d.stat().st_ctime)
        instance_folder = most_recent_dir
    instance_folder = instance_folder.resolve()
    gif_dir = instance_folder / "gif-build"
    eps_dir = gif_dir / "eps"
    png_dir = gif_dir / "png"
    outpath = instance_folder / "animation.gif"

    gif_builder = GifBuilder(eps_dir, png_dir)
    # You can comment out the next line if you already have PNGs prepared.
    gif_builder.convert_eps_files()
    gif_builder.make_gif(outpath)


if __name__ == "__main__":
    parser = ArgumentParser(description="Make a gif from EPS or PNG images.")
    parser.add_argument("--folderpath", type=str, help="Path to the folder.")

    args = parser.parse_args()
    instance_folder = Path(args.folderpath) if args.folderpath else args.folderpath
    make_gif_from_images(instance_folder)
