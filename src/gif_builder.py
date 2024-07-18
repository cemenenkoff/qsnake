from pathlib import Path

from PIL import EpsImagePlugin, Image
from tqdm import tqdm

# This script requires a separate installation of Ghostscript!
EpsImagePlugin.gs_windows_binary = r"C:\Program Files\gs\gs9.55.0\bin\gswin64c"


class GifBuilder:
    def __init__(self, eps_dir: Path, png_dir: Path) -> None:
        """Initialize a `GifBuilder`.

        Args:
            eps_dir (Path): Directory housing the eps image files.
            png_dir (Path): The directory to house the generated png files.
        """
        self.eps_dir = eps_dir
        self.png_dir = png_dir

    def _eps2png(self, eps_fpath: Path) -> None:
        """Convert an eps file into a png.

        Args:
            eps_fpath (Path): Path to the eps file.
        """
        png_fname = eps_fpath.name.replace(".eps", ".png")
        im = Image.open(eps_fpath)
        fig = im.convert("RGBA")
        fig.save(self.png_dir / png_fname)
        im.close()

    def convert_eps_files(self) -> None:
        """Convert all eps files in a directory into png format all at once."""
        if not self.png_dir.exists():
            self.png_dir.mkdir(parents=True)
        eps_fpaths = [eps_fpath for eps_fpath in self.eps_dir.glob("*.eps")]
        print("Converting eps files to png (this may take some time)...")
        for eps_fpath in tqdm(eps_fpaths):
            self._eps2png(Path(eps_fpath))

    def make_gif(self, outpath: Path = Path.cwd() / "training_montage.gif") -> None:
        """Create an animated gif from an ordered series of png files.

        Args:
            outpath (Path, optional): The desired output path for the finished gif.
                Defaults to Path.cwd()/"training_montage.gif".
        """
        png_fpaths = [png_file for png_file in self.png_dir.glob("*.png")]
        image_list = []
        # Note that we open and close a temp file to avoid potentially opening several
        # thousand files at once.
        for png_file in tqdm(png_fpaths):
            temp = Image.open(png_file)
            keep = temp.copy()
            image_list.append(keep)
            temp.close()
        if not outpath.exists():
            outpath.parent.mkdir(exist_ok=True, parents=True)
        im = image_list.pop(0)  # The gif builds by appending to a base image.
        print("Creating animated gif (this may take some time)...")
        try:
            # A note on animation speed: A duration of 20ms means 50fps, which is the
            # max for what most web browsers support. Image viewers like IrfanView can
            # play gifs up to 100fps.
            im.save(
                outpath, save_all=True, append_images=image_list, duration=20, loop=0
            )  # 0 makes the gif loop indefinitely.
        except Exception as exc:
            print(f"ERROR: Animated gif creation failed: {exc}")
        print(f"Animated gif exported to: {outpath}")
