import turtle
from pathlib import Path
from PIL import Image
from datetime import datetime
ts = datetime.now().strftime('%c').replace(' ','-').replace(':','-')

PROJECT_ROOT_DIR = '.' #the current working directory
IMAGES_PATH = Path(PROJECT_ROOT_DIR)/'figures'/f'gif_{ts}'
IMAGES_PATH.mkdir(exist_ok=True, parents=True)

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
        outpath = Path(IMAGES_PATH)/eps_fname
        turtle.getcanvas().postscript(file=outpath)
        im = Image.open(outpath)
        fig = im.convert('RGBA')
        png_fname = eps_fname.replace('.eps','.png')
        fig.save(Path(IMAGES_PATH)/png_fname)

    def run_game(self):
        self.t.forward(self.s)
        output_file = f'{self.i}.eps'
        self.win.update()
        self.eps2png(output_file)
        # turtle.getcanvas().postscript(file=Path(IMAGES_PATH)/output_file)
        self.t.left(90)
        self.win.update()
        self.t.forward(self.s)
        self.win.update()
        self.t.left(90)
        self.win.update()
        self.t.forward(self.s)
        self.win.update()
        self.t.left(90)
        self.win.update()
        self.t.forward(self.s)
        self.win.update()
        self.t.left(90)
        self.win.update()

test = Animation(100)
test.run_game()