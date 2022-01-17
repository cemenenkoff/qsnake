from gif_creator import GifBuilder
from pathlib import Path

# After running explore.py with `"save_for_gif": true`, this script allows us
# to loop through the saved eps files and turn them into an animated gif.

config = {
    'eps_dir':Path(r".\figures\gif-attempt-1-SunJan16-212029-default-50ep-256batch\gif-build\eps"),
    'png_dir':Path(r".\figures\gif-attempt-1-SunJan16-212029-default-50ep-256batch\gif-build\png"),
    'outpath':Path(r".\test.gif")
}

bob_the_builder = GifBuilder(config['eps_dir'], config['png_dir'])
#bob_the_builder.convert_eps_files() # Toggle this off if you already have pngs.
bob_the_builder.make_gif(outpath=config['outpath'])