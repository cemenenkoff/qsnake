import turtle
from pathlib import Path
from PIL import Image
from datetime import datetime
ts = datetime.now().strftime('%c').replace(' ','-').replace(':','-')

import PIL
from PIL import EpsImagePlugin
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.55.0\bin\gswin64c'
PROJECT_ROOT_DIR = '.' #the current working directory
IMAGES_DIR = Path(PROJECT_ROOT_DIR)/'figures'/f'gif_{ts}'
IMAGES_DIR.mkdir(exist_ok=True, parents=True)
EPS_DIR = Path(IMAGES_DIR)/'eps'
EPS_DIR.mkdir(exist_ok=True, parents=True)
PNG_DIR = Path(IMAGES_DIR)/'png'
PNG_DIR.mkdir(exist_ok=True, parents=True)

class Animation():
    def __init__(self, s):
        self.s=s
        self.win = turtle.Screen()
        self.win.title('test')
        self.win.bgcolor('white')
        # self.win.tracer(0)
        self.win.setup(width=400+32, height=400+32) # +32 is an eyeballed frame adjustment.
        self.t = turtle.Turtle()
        self.t.speed(1)
        self.i=0

    def eps2png(self, eps_fname):
        eps_out = EPS_DIR/eps_fname
        turtle.getcanvas().postscript(file=eps_out)
        im = Image.open(eps_out)
        fig = im.convert('RGBA')
        png_fname = eps_fname.replace('.eps','.png')
        fig.save(PNG_DIR/png_fname)

    def step(self):
        output_file = f'{self.i}.eps'
        self.win.update()
        self.eps2png(output_file)
        self.i+=1

    def run_game(self):
        self.t.forward(self.s)
        self.step()
        self.t.left(90)
        self.step()
        self.t.forward(self.s)
        self.step()
        self.t.left(90)
        self.step()
        self.t.forward(self.s)
        self.step()
        self.t.left(90)
        self.step()
        self.t.forward(self.s)
        self.step()
        self.t.left(90)
        self.win.update()

    def make_gif(self):
        png_files = PNG_DIR.glob('*.png')
        image_list = [PIL.Image.open(str(_)) for _ in png_files]
        im = image_list.pop(0)
        im.save(IMAGES_DIR/'training.gif', save_all=True, append_images=image_list)

test = Animation(100)
test.run_game()
test.make_gif()