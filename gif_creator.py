from PIL import EpsImagePlugin, Image
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

# This script requires a separate installation of Ghostscript!
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.55.0\bin\gswin64c'

def eps2png(eps_fpath:Path, png_dir):
    im = Image.open(eps_fpath)
    fig = im.convert('RGBA')
    png_fname = eps_fpath.name.replace('.eps','.png')
    fig.save(png_dir/png_fname)

def make_gif(eps_dir, png_dir, outpath=None):
    eps_fpaths_gen = eps_dir.glob('*.eps')
    eps_fpaths = [fpath for fpath in eps_fpaths_gen]
    eps_fpaths = natsorted(eps_fpaths)
    for eps_fpath in tqdm(eps_fpaths):
        eps2png(eps_fpath, png_dir)
    png_files = png_dir.glob('*.png')
    image_list = [Image.open(str(_)) for _ in png_files]
    im = image_list.pop(0)
    outpath = 'training_montage.gif' if not outpath else outpath
    im.save(outpath, save_all=True, append_images=image_list)