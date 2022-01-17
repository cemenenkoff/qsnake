import turtle
import random
import time
import math
import gym
import sys
from pathlib import Path

class Snake(gym.Env):
    FONT_FAM = 'Courier'
    FONT_SIZE = 18
    FONT_ALIGN = 'center'
    FONT_STYLE = 'normal'
    HEAD_SIZE = 20              # side length of the square snake head in pixels
    HEIGHT = WIDTH = 20         # number of snake heads that can stack vertically up (or horizontally along) the square screen
    PIXEL_H = HEAD_SIZE*HEIGHT  # height of the screen in pixels
    PIXEL_W = HEAD_SIZE*WIDTH   # width of the screen in pixels
    SLEEP = 0.1                 # seconds to wait between steps for human players
    GAME_TITLE = 'Snake'
    BG_COLOR = tuple(i/255.0 for i in (242,225,242)) # lavender is gentler than bright white
    SNAKE_SHAPE = 'square'
    SNAKE_COLOR = 'green'
    SNAKE_SPEED = 'fastest'
    SNAKE_START_X = HEAD_SIZE*0 # Note that the origin of our x-y coordinate frame is in the center of the screen.
    SNAKE_START_Y = HEAD_SIZE*0 # These starting coordinates are in units of pixels.
    APPLE_SHAPE = 'circle'
    APPLE_COLOR = 'red'

    def __init__(self, config):
        super(Snake, self).__init__() # Initialize an Env class from OpenAI's gym.

        self.state_definition_type = config['params']['state_definition_type'] # what defines our 12-dimensional state vector
        self.human = config['human']
        self.save_for_gif = config['save_for_gif']
        self.eps_dir = config.get('eps_dir')
        self.step_number = 0
        self.episode_number = 0
        self.done = False # whether or not the game is over
        self.action_space = 4 # The dimension of the action space is 4 (up, right, down, left).
        self.state_space = 12 # The state_space (a.k.a. observation_space) is 12 because our state vector has 12 elements.
        self.reward=0
        self.total=0
        self.maximum=0

        # Create the background Screen in which the snake hunts for the apple.
        self.win = turtle.Screen()
        self.win.title(self.GAME_TITLE)
        self.win.bgcolor(*self.BG_COLOR)
        self.win.tracer(0)
        self.win.setup(width=self.PIXEL_W+32, height=self.PIXEL_H+32) # +32 is an eyeballed frame adjustment.

        # Create the snake itself which is initially a head and an (invisible) dummy body chunk.
        self.head = turtle.Turtle()
        self.head.shape(self.SNAKE_SHAPE)
        # The default turtlesize of (1.0, 1.0, 1.0) means 20-pixels width, 20-pixels height, and 1-width for the shape's outline.
        self.head.turtlesize(*tuple(self.HEAD_SIZE*num/20 for num in self.head.turtlesize()))
        self.head.speed(self.SNAKE_SPEED)
        self.head.penup() # Pull the pen up -- no drawing when moving.
        self.head.color(self.SNAKE_COLOR)
        self.head.goto(self.SNAKE_START_X, self.SNAKE_START_Y)
        self.head.direction = 'stop' # The possible directions are 'up', 'right', 'down', 'left', or 'stop'.
        self.body = [] # The snake body is a list of turtle objects that increments as it eats.
        self.append_body_chunk() # Instantiate the snake with a dummy body chunk. This makes tracking how the body moves a bit easier.

        # Create the apple that the snake should hunt.
        self.apple = turtle.Turtle()
        self.apple.shape(self.APPLE_SHAPE)
        self.apple.color(self.APPLE_COLOR)
        self.apple.turtlesize(*tuple(self.HEAD_SIZE*num/20 for num in self.apple.turtlesize()))
        self.apple.penup()
        self.spawn_apple(first=True)

        # Calculate the Pythagorean distance between the apple and the head of the snake.
        self.dist = math.sqrt((self.head.xcor()-self.apple.xcor())**2 + (self.head.ycor()-self.apple.ycor())**2)

        # Create a text scoreboard that will periodically update with information.
        self.score = turtle.Turtle()
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, int(0.75*self.PIXEL_H/2))
        self.score.write(f'Total: {self.total}\tBest: {self.maximum}', align=self.FONT_ALIGN, font=(self.FONT_FAM, self.FONT_SIZE, self.FONT_STYLE))

        # Define user controls.
        self.win.listen() # win.listen sets the focus on the TurtleScreen in order to collect key events.
        self.win.onkeypress(self.move_up, 'Up')
        self.win.onkeypress(self.move_down, 'Down')
        self.win.onkeypress(self.move_left, 'Left')
        self.win.onkeypress(self.move_right, 'Right')

    def move_head(self):
        '''
        changes the snake head's position according to its current direction
        '''
        if self.head.direction == 'up':
            y = self.head.ycor()
            self.head.sety(y + self.HEAD_SIZE) # Remember that sety and setx are measured in pixels.
        elif self.head.direction == 'down':
            y = self.head.ycor()
            self.head.sety(y - self.HEAD_SIZE)
        elif self.head.direction == 'left':
            x = self.head.xcor()
            self.head.setx(x - self.HEAD_SIZE)
        elif self.head.direction == 'right':
            x = self.head.xcor()
            self.head.setx(x + self.HEAD_SIZE)
        else: # This means self.head.direction == 'stop', so reset the reward.
            self.reward = 0

    def move_body(self): # Remember the 0th index is the first (dummy) body chunk and not the snake's head.
        '''
        moves the snakes body, following the head's lead
        '''
        for i, chunk in reversed(list(enumerate(self.body))): # e.g. 10, 9, 8, 7, ...
            if i != 0:
                x = self.body[i-1].xcor() # Get the coordinates of the chunk before.
                y = self.body[i-1].ycor()
                self.body[i].goto(x, y) # This chunk moves to wherever the chunk to the left (in the body array) was.
            else: # The leftmost (in the array) body piece moves to the head's last position.
                self.body[i].goto(self.head.xcor(), self.head.ycor())

    def move_up(self):
        self.head.direction = 'up' if self.head.direction != 'down' else self.head.direction
    def move_down(self):
        self.head.direction = 'down' if self.head.direction != 'up' else self.head.direction
    def move_left(self):
        self.head.direction = 'left' if self.head.direction != 'right' else self.head.direction
    def move_right(self):
        self.head.direction = 'right' if self.head.direction != 'left' else self.head.direction

    def append_body_chunk(self):
        '''
        appends a chunk to elongate the snake's body after it successfully eats an apple
        '''
        chunk = turtle.Turtle()
        chunk.speed(self.SNAKE_SPEED)
        chunk.shape(self.SNAKE_SHAPE)
        # Scale the body chunks to be 80% as large as the head.
        chunk.turtlesize(*tuple(0.8*self.HEAD_SIZE*num/20 for num in self.head.turtlesize()))
        chunk.color(self.SNAKE_COLOR)
        chunk.penup()
        self.body.append(chunk) # Incrementally add additional turtles of the same shape to a running list.

    def update_score(self):
        '''
        increments and updates the scoreboard
        '''
        self.total += 1
        self.maximum = self.total if self.total >= self.maximum else self.maximum
        self.score.clear()
        self.score.write(f"Total: {self.total}\tBest: {self.maximum}", align=self.FONT_ALIGN, font=(self.FONT_FAM, self.FONT_SIZE, self.FONT_STYLE))

    def get_random_coordinates(self):
        '''
        returns coordinates in units of HEAD_SIZE
        '''
        x = random.randint(-self.WIDTH/2, self.WIDTH/2)
        y = random.randint(-self.HEIGHT/2, self.HEIGHT/2)
        return x, y

    def spawn_apple(self, first=False):
        '''
        spawns the apple at a random location on the screen that is not inside the snake
        '''
        while True:
            self.apple.x, self.apple.y = self.get_random_coordinates()
            self.apple.goto(round(self.apple.x*self.HEAD_SIZE), round(self.apple.y*self.HEAD_SIZE))
            if not self.is_eating_apple(): # Make sure the apple doesn't spawn in the snake itself.
                break
        if not first:
            self.update_score()
            self.append_body_chunk()
        return True

    def reset_score(self):
        '''
        resets the scoreboard (keeping the best score)
        '''
        self.score.clear()
        self.total = 0
        self.score.write(f"Total: {self.total}\tBest: {self.maximum}", align=self.FONT_ALIGN, font=(self.FONT_FAM, self.FONT_SIZE, self.FONT_STYLE))

    def get_distance_to_apple(self):
        '''
        calculates the straight-line distance from the snake's head to the apple
        '''
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.head.xcor()-self.apple.xcor())**2 + (self.head.ycor()-self.apple.ycor())**2)

    def is_eating_body(self):
        '''
        checks to see if the snake is eating its body
        '''                                                                       # [o][o]
        if any([body.distance(self.head) < self.HEAD_SIZE for body in self.body[3:]]): # [o][<] You need a length of at leaset 4 to eat yourself.
            self.reset_score()
            return True

    def is_eating_apple(self):
        '''
        checks to see if the snake is eating an apple
        '''
        if len(self.body)>0:
            if any([chunk.distance(self.apple) < self.HEAD_SIZE for chunk in self.body]):
                return True
        if self.head.distance(self.apple) < self.HEAD_SIZE:
            return True

    def is_hitting_wall(self):
        '''
        checks to see if the snake is hitting a wall
        '''
        if any([self.head.xcor() >  self.PIXEL_W/2, self.head.xcor() < -self.PIXEL_W/2,
                self.head.ycor() >  self.PIXEL_H/2, self.head.ycor() < -self.PIXEL_H/2]):
            self.reset_score()
            return True

    def reset(self):
        '''
        Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        '''
        if self.human:
            time.sleep(1)
        for chunk in self.body: # Hide the body.
            chunk.hideturtle()
        self.body = []
        self.head.goto(self.SNAKE_START_X, self.SNAKE_START_Y)
        self.head.direction = 'stop' # Reinitialize the starting direction in pause mode.
        self.reward=self.total=0
        self.done = False
        return self.get_state()

    def save_eps(self):
        '''
        saves the current frame as an eps file
        '''
        eps_fname = f'ep{self.episode_number:09d}-step{self.step_number:09d}.eps'
        eps_outpath = self.eps_dir/eps_fname if isinstance(self.eps_dir, Path) else eps_fname
        turtle.getcanvas().postscript(file=eps_outpath, colormode='color')

    def run_game(self):
        '''
        runs the main snake game loop
        '''
        reward_given = False # Flip this to True if the snake gains a reward during a time step.
        try:
            self.win.update()
            self.move_head()
            if self.head.distance(self.apple) < self.HEAD_SIZE:
                self.spawn_apple() # If we managed to munch an apple, respawn the apple at a new location.
                self.reward = 10
                reward_given = True
            self.move_body() # After the snake head moves, update the body.
            self.get_distance_to_apple() # Once the head and body have both moved, recalculate the distance from the head to the apple.
        except: # If we throw an error in an attempt to exit the game, just exit.
            sys.exit()

        if self.is_eating_body(): # Check to see if the snake is eating itself.
            self.reward = -100 # Enforce a large disincentive for eating your own body.
            reward_given = self.done = True
            if self.human:
                self.reset()
        if self.is_hitting_wall(): # Check to see if the snake hit a wall in this frame.
            self.reward = -100 # Eating your own body is just as bad as running into a wall.
            reward_given = self.done = True
            if self.human:
                self.reset()
        if not reward_given:
            self.reward=1 if self.dist < self.prev_dist else -1
        if self.human:
            time.sleep(self.SLEEP)
        if self.save_for_gif:
            self.save_eps()

    def step(self, action, episode_number=None, step_number=None):
        '''
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        '''
        if action == 0: self.move_up()
        if action == 1: self.move_down()
        if action == 2: self.move_left()
        if action == 3: self.move_right()
        if isinstance(episode_number, int) and isinstance(step_number, int) and step_number==0:
            self.episode_number = episode_number
        if isinstance(step_number, int):
            self.step_number = step_number
        self.run_game()
        return self.get_state(), self.reward, self.done, {}

    def get_state(self):
        '''
        obtains the 12-dimensional state of the snake
        '''
        # Define the coordinates of the snake head in units of HEAD_SIZE.
        self.head.x, self.head.y = self.head.xcor()/self.WIDTH, self.head.ycor()/self.HEIGHT

        # Scale the coordinates of the snake head down to range from 0 to 1.
        # This requires shifting the origin to the left by half the width of the 1-length x-interval,
        # and also shifting down by half the height of the 1-length y-interval.
        self.head.xsc, self.head.ysc = self.head.x/self.WIDTH+0.5, self.head.y/self.HEIGHT+0.5

        # Scale the coordinates of the apple to range between 0 and 1 in the same way.
        self.apple.xsc, self.apple.ysc = self.apple.x/self.WIDTH+0.5, self.apple.y/self.HEIGHT+0.5

        # Check to see which cardinal directions point toward the apple.
        apple_above=1 if self.head.y < self.apple.y else 0
        apple_below=1 if self.head.y > self.apple.y else 0
        apple_left =1 if self.head.x < self.apple.x else 0
        apple_right=1 if self.head.x > self.apple.x else 0

        # Check to if the head is adjacent to a wall and in what cardinal direction.
        wall_above=1 if  self.HEIGHT/2 - 1 <= self.head.y and self.head.y <=  self.HEIGHT/2     else 0 # within one HEAD_SIZE unit below the top wall
        wall_below=1 if -self.HEIGHT/2     <= self.head.y and self.head.y <= -self.HEIGHT/2 + 1 else 0 # within one HEAD_SIZE unit to the left of the right wall
        wall_left =1 if -self.WIDTH /2     <= self.head.x and self.head.x <= -self.WIDTH /2 + 1 else 0 # within one HEAD_SIZE unit to the right of the left wall
        wall_right=1 if  self.WIDTH /2 - 1 <= self.head.x and self.head.x <=  self.WIDTH /2     else 0 # within one HEAD_SIZE unit above the bottom wall

        # Check to see if some of the snake's body chunks are adjacent to the head.
        # Here are some example states where ^ is the head and . is the tail:
        #   [0][1][0]     [.][1][0]       [^]
        #   [1][^][1]        [^][1]       [0]
        #   [.][1][0]        [1][0]       [.]
        body_above=body_below=body_left=body_right=False
        if len(self.body) > 3:          # [0][1]
            for chunk in self.body[3:]: # [.][<] You need a length of at least 4 to eat yourself.
                if chunk.distance(self.head) == self.HEAD_SIZE:
                    if chunk.ycor() < self.head.ycor():
                        body_below = True
                    if chunk.ycor() > self.head.ycor():
                        body_above = True
                    if chunk.xcor() < self.head.xcor():
                        body_left = True
                    if chunk.xcor() > self.head.xcor():
                        body_right = True

        # Check to see if a wall OR a body chunk (an obstacle) is adjacent to the head.
        obstacle_above=1 if wall_above or body_above else 0
        obstacle_below=1 if wall_below or body_below else 0
        obstacle_left =1 if wall_left  or  body_left else 0
        obstacle_right=1 if wall_right or body_right else 0

        # One-hot encode the head's direction attribute.
        direction_up   =1 if self.head.direction==   'up' else 0
        direction_down =1 if self.head.direction== 'down' else 0
        direction_left =1 if self.head.direction== 'left' else 0
        direction_right=1 if self.head.direction=='right' else 0

        if self.state_definition_type == 'apple_coords': # Let the agent receive direct knowledge of where the apple is and where the head is.
            state = [self.apple.xsc, self.apple.ysc,  self.head.xsc,   self.head.ysc,
                     obstacle_above, obstacle_below,  obstacle_left,  obstacle_right,
                       direction_up, direction_down, direction_left, direction_right]
        elif self.state_definition_type == 'no_dir': # Don't let the agent know the direction in which the head is moving.
            state = [   apple_above,    apple_below,     apple_left,     apple_right,
                     obstacle_above, obstacle_below,  obstacle_left,  obstacle_right,
                                  0,              0,              0,               0]
        elif self.state_definition_type == 'no_body': # Remove body chunks from the definition of the obstacle substates.
            state = [   apple_above,    apple_below,     apple_left,     apple_right,
                         wall_above,     wall_right,     wall_below,       wall_left,
                       direction_up, direction_down, direction_left, direction_right]
        else: # The 'default' state includes apple direction, obstacle adjacency, and head direction information.
            state = [   apple_above,    apple_below,     apple_left,     apple_right,
                     obstacle_above, obstacle_below,  obstacle_left,  obstacle_right,
                       direction_up, direction_down, direction_left, direction_right]
        return state
