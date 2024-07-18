from pathlib import Path

from gif_builder import GifBuilder


def main() -> None:
    """Loop through eps files and turn them into an animated gif.

    This script is intended to run after running `explore.py` with:
        `"save_for_gif": true`
    specified within the associated `config.json`.
    """
    figures_dir = Path("figures")
    instance_folder = "test-animation-2-MonJan17-095750-default-50ep-256batch"
    instance_dir = figures_dir / instance_folder
    config = {
        "eps_dir": instance_dir / "gif-build" / "eps",
        "png_dir": instance_dir / "gif-build" / "png",
        "outpath": instance_dir / "animation.gif",
    }
    bob_the_builder = GifBuilder(config["eps_dir"], config["png_dir"])
    # bob_the_builder.convert_eps_files() # Toggle this off if you already have pngs.
    bob_the_builder.make_gif(outpath=config["outpath"])


if __name__ == "__main__":
    main()
