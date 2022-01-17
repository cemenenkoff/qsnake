from PIL import EpsImagePlugin, Image
from pathlib import Path
from tqdm import tqdm

# This script requires a separate installation of Ghostscript!
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.55.0\bin\gswin64c'

class GifBuilder:
    def __init__(self, eps_dir:Path, png_dir:Path):
        self.eps_dir = eps_dir
        self.png_dir = png_dir

    def _eps2png(self, eps_fpath:Path):
        png_fname = eps_fpath.name.replace('.eps','.png')
        im = Image.open(eps_fpath)
        fig = im.convert('RGBA')
        fig.save(self.png_dir/png_fname)
        im.close()

    def convert_eps_files(self):
        if not self.png_dir.exists:
            self.png_dir.mkdir(parents=True)
        for eps_fpath in tqdm(self.eps_dir.glob('*.eps')):
            self._eps2png(Path(eps_fpath))

    def make_gif(self, outpath:Path=Path.cwd()/'training_montage.gif'):
        png_fpaths = [png_file for png_file in self.png_dir.glob('*.png')]
        image_list = []
        # Note that we open and close a temp file to avoid potentially opening
        # several thousand files at once.
        for png_file in tqdm(png_fpaths):
            temp = Image.open(png_file)
            keep = temp.copy()
            image_list.append(keep)
            temp.close()
        if outpath!=Path.cwd()/'training_montage.gif':
            outpath.parent.mkdir(exist_ok=True, parents=True)
        # The 1st image is the base; the rest get appended as additional frames.
        im = image_list.pop(0)
        print('creating gif (this may take some time)')
        try:
            # A note on animation speed: a duration of 20ms means 50fps, which
            # is what most web browsers support. Image viewers like IrfanView
            # can play gifs up to 100fps.
            im.save(outpath, save_all=True, append_images=image_list,
                    duration=20)
        except Exception as e:
            print(f'ERROR: gif creation failed: {e}')
        print(f'gif exported to {outpath}')