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
        self.eps_dir = eps_dir.resolve()
        self.png_dir = png_dir.resolve()

    def convert_eps_to_png(self, eps_path: Path) -> None:
        """Convert an EPS file to a PNG.

        Args:
            eps_path (Path): Path to the EPS file.
        """
        png_fname = eps_path.with_suffix(".png").name
        with Image.open(eps_path) as img:
            img = img.convert("RGBA")
            img.save(self.png_dir / png_fname)

    def convert_eps_files(self) -> None:
        """Convert all eps files in a directory into png format all at once."""
        if not self.png_dir.exists():
            self.png_dir.mkdir(parents=True)
        eps_fpaths = [eps_path for eps_path in self.eps_dir.glob("*.eps")]
        print("Converting eps files to png (this may take some time)...")
        for eps_path in tqdm(eps_fpaths):
            self.convert_eps_to_png(eps_path)

    def make_gif(self, outpath: Path = Path.cwd() / "training_montage.gif") -> None:
        """Create an animated gif from an ordered series of png files.

        Args:
            outpath (Path, optional): The desired output path for the finished gif.
                Defaults to Path.cwd()/"training_montage.gif".
        """
        pngs = [png for png in self.png_dir.glob("*.png")]
        frames = []
        # Note that we open and close a temp file to avoid potentially opening several
        # thousand files at once.
        for png_file in tqdm(pngs):
            temp = Image.open(png_file)
            keep = temp.copy()
            temp.close()
            frames.append(keep)
        if not outpath.exists():
            outpath.parent.mkdir(exist_ok=True, parents=True)
        img = frames.pop(0)  # The gif builds by appending to a base image.
        print("Creating animated gif (this may take some time)...")
        try:
            # A note on animation speed: A duration of 20ms means 50fps, which is the
            # max for what most web browsers support. Image viewers like IrfanView can
            # play gifs up to 100fps.
            img.save(
                outpath,
                save_all=True,
                append_images=frames,
                duration=20,
                loop=0,  # 0 makes the gif loop indefinitely.
                transparency=0,
                disposal=2,  # Remove lingering frames. See: https://tinyurl.com/mrxn7f4f
            )
        except Exception as exc:
            print(f"ERROR: Animated gif creation failed: {exc}")
        print(f"Animated gif exported to: {outpath}")
