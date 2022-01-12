import turtle
import random
import time
import math
import gym
import sys

HUMAN = True
FONT_FAM = 'Courier'
FONT_SIZE = 18
FONT_ALIGN = 'center'
FONT_STYLE = 'normal'
HEAD_SIZE = 20              # side length of the square snake head in pixels
HEIGHT = WIDTH = 20         # number of snake heads that can stack vertically up (or horizontally along) the square screen
PIXEL_H = HEAD_SIZE*HEIGHT  # height of the screen in pixels
PIXEL_W = HEAD_SIZE*WIDTH   # width of the screen in pixels
SLEEP = 0.2                 # seconds to wait between steps
GAME_TITLE = 'Snake'
BG_COLOR = tuple(i/255.0 for i in (242,225,242)) # lavender is gentler than bright white
SNAKE_SHAPE = 'square'
SNAKE_COLOR = 'green'
SNAKE_SPEED = 'normal'
SNAKE_START_X = HEAD_SIZE*0 # Note that the origin of our x-y coordinate frame is in the center of the screen.
SNAKE_START_Y = HEAD_SIZE*0 # These starting coordinates are in units of pixels.
APPLE_SHAPE = 'circle'
APPLE_COLOR = 'red'

class Snake(gym.Env):

    def __init__(self, human=False, state_definition_type='default'):
        super(Snake, self).__init__() # Initialize an Env class from OpenAI's gym.

        self.done = False
        self.action_space = 4 # The dimension of the action space is 4 (up, right, down, left).
        self.state_space = 12
        self.reward=self.total=self.maximum=0
        self.human = human
        self.state_definition_type = state_definition_type

        # Create the background in which the snake hunts for the apple.
        self.win = turtle.Screen()
        self.win.title(GAME_TITLE)
        self.win.bgcolor(*BG_COLOR)
        self.win.tracer(0)
        self.win.setup(width=PIXEL_W+32, height=PIXEL_H+32) # +32 is a eyeballed frame adjustment.

        # Create the snake itself which is initially a head and an (invisible) dummy body chunk.
        self.snake = turtle.Turtle()
        self.snake.shape(SNAKE_SHAPE)
        # The default turtlesize of (1.0, 1.0, 1.0) means 20-pixels width , 20-pixels height, and 1-width for the shape's outline.
        self.snake.turtlesize(*tuple(HEAD_SIZE*num/20 for num in self.snake.turtlesize()))
        self.snake.speed(SNAKE_SPEED)
        self.snake.penup() # Pull the pen up -- no drawing when moving.
        self.snake.color(SNAKE_COLOR)
        self.snake.goto(SNAKE_START_X, SNAKE_START_Y)
        self.snake.direction = 'stop' # Possible states are 'up', 'right', 'down', 'left', or 'stop'
        # snake body, add first element (for location of snake's head)
        self.snake_body = [] # The snake body is a list of turtle objects that increments as it eats.
        self.append_body_chunk() # Instantiate the snake with a dummy body chunk (the body anchor piece?).
        # Having a dummy chunk which shares the same coordinates as the head makes tracking how the body moves a bit easier.

        # Create the apple that the snake should hunt.
        self.apple = turtle.Turtle()
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.turtlesize(*tuple(HEAD_SIZE*num/20 for num in self.apple.turtlesize()))
        self.apple.penup()
        self.spawn_apple(first=True) # Create the apple at a set of random on-screen coordinates for the first time.

        # Calculate the Pythagorean distance between the apple and the head of the snake.
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)

        # Create a text scoreboard that will periodically update with information.
        self.score = turtle.Turtle()
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, int(0.75*PIXEL_H/2))
        self.score.write(f'Total: {self.total}\tHighest: {self.maximum}', align=FONT_ALIGN, font=(FONT_FAM, FONT_SIZE, FONT_STYLE))

        # Define user controls.
        self.win.listen() # win.listen sets the focus on the TurtleScreen in order to collect key-events.
        self.win.onkeypress(self.go_up, 'Up')
        self.win.onkeypress(self.go_right, 'Right')
        self.win.onkeypress(self.go_down, 'Down')
        self.win.onkeypress(self.go_left, 'Left')

    def random_coordinates(self):
        '''
        returns coordinates in units of HEAD_SIZE
        '''
        x = random.randint(-WIDTH/2, WIDTH/2)
        y = random.randint(-HEIGHT/2, HEIGHT/2)
        return x, y

    def move_snake(self):
        '''
        change the snake head's position according to its current direction
        '''
        if self.snake.direction == 'up':
            y = self.snake.ycor()
            self.snake.sety(y + HEAD_SIZE)
        elif self.snake.direction == 'right':
            x = self.snake.xcor()
            self.snake.setx(x + HEAD_SIZE)
        elif self.snake.direction == 'down':
            y = self.snake.ycor()
            self.snake.sety(y - HEAD_SIZE)
        elif self.snake.direction == 'left':
            x = self.snake.xcor()
            self.snake.setx(x - HEAD_SIZE)
        else: # This means self.snake.direction == 'stop', so reset the reward.
            self.reward = 0

    def go_up(self):
        self.snake.direction = 'up' if self.snake.direction != 'down' else self.snake.direction
    def go_right(self):
        self.snake.direction = 'right' if self.snake.direction != 'left' else self.snake.direction
    def go_down(self):
        self.snake.direction = 'down' if self.snake.direction != 'up' else self.snake.direction
    def go_left(self):
        self.snake.direction = 'left' if self.snake.direction != 'right' else self.snake.direction

    def spawn_apple(self, first=False):
        '''
        spawns the apple at a random location on the screen that is not inside the snake
        '''
        while True:
            self.apple.x, self.apple.y = self.random_coordinates()
            self.apple.goto(round(self.apple.x*HEAD_SIZE), round(self.apple.y*HEAD_SIZE))
            if not self.is_eating_apple(): # Make sure the apple doesn't spawn in the snake itself.
                break
        if not first:
            self.update_score()
            self.append_body_chunk()
        return True

    def update_score(self):
        '''
        increments and updates the scoreboard
        '''
        self.total += 1
        self.maximum = self.total if self.total >= self.maximum else self.maximum
        self.score.clear()
        self.score.write(f"Total: {self.total}\tHighest: {self.maximum}", align=FONT_ALIGN, font=(FONT_FAM, FONT_SIZE, FONT_STYLE))

    def reset_score(self):
        '''
        resets the scoreboard (keeping the highest score)
        '''
        self.score.clear()
        self.total = 0
        self.score.write(f"Total: {self.total}\tHighest: {self.maximum}", align=FONT_ALIGN, font=(FONT_FAM, FONT_SIZE, FONT_STYLE))

    def append_body_chunk(self):
        '''
        appends a chunk to elongate the snake's body after it successfully eats an apple
        '''
        chunk = turtle.Turtle()
        chunk.speed(SNAKE_SPEED)
        chunk.shape(SNAKE_SHAPE)
        # Scale the body chunks to be 80% as large as the head.
        chunk.turtlesize(*tuple(0.8*HEAD_SIZE*num/20 for num in self.snake.turtlesize()))
        chunk.color(SNAKE_COLOR)
        chunk.penup()
        self.snake_body.append(chunk) # Incrementally add additional turtles of the same shape to a running list.

    def move_snakebody(self): # Remember the 0th index is the first (dummy) body chunk and not the snake's head.
        '''
        moves the snakes body
        '''
        for i, chunk in reversed(list(enumerate(self.snake_body))): # e.g. 10, 9, 8, 7, ...
            if i != 0:
                x = self.snake_body[i-1].xcor() # Get the coordinates of the chunk before.
                y = self.snake_body[i-1].ycor()
                self.snake_body[i].goto(x, y) # This chunk moves to wherever the chunk to the left (in the body array) was.
            else: # The leftmost (in the array) body piece moves to the head's last position.
                self.snake_body[i].goto(self.snake.xcor(), self.snake.ycor())

    def get_distance_to_apple(self):
        '''
        calculates the straight-line distance from the snake's head to the apple
        '''
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + (self.snake.ycor()-self.apple.ycor())**2)

                             # [o][o]
    def is_eating_body(self):# [o][<] You need a length of at leaset 4 to eat yourself.
        '''
        checks to see if the snake is eating its body
        '''
        if any([body.distance(self.snake) < HEAD_SIZE for body in self.snake_body[3:]]):
            self.reset_score()
            return True

    def is_eating_apple(self):
        '''
        checks to see if the snake is eating an apple
        '''
        if len(self.snake_body)>0:
            if any([body.distance(self.apple) < HEAD_SIZE for body in self.snake_body]):
                return True
        if self.snake.distance(self.apple) < HEAD_SIZE:
            return True

    def is_hitting_wall(self):
        '''
        checks to see if the snake is hitting a wall
        '''
        if any([self.snake.xcor() >  PIXEL_W/2, self.snake.xcor() < -PIXEL_W/2,
                self.snake.ycor() >  PIXEL_H/2, self.snake.ycor() < -PIXEL_H/2]):
            self.reset_score()
            return True

    def reset(self):
        '''
        resets the game with the snake in the center
        '''
        if self.human:
            time.sleep(1)
        for body in self.snake_body: # Hide the body.
            body.hideturtle()
        self.snake_body = []
        self.snake.goto(SNAKE_START_X, SNAKE_START_Y)
        self.snake.direction = 'stop' # Reinitialize the starting direction in pause mode.
        self.reward=self.total=0
        self.done = False
        return self.get_state()

    def run_game(self):
        '''
        runs the main snake game loop
        '''
        reward_given = False # Flip this to True if the snake gains a reward during a time step.
        try:
            self.win.update()
            self.move_snake()
            if self.snake.distance(self.apple) < HEAD_SIZE:
                self.spawn_apple() # If we managed to munch an apple, respawn the apple at a new location.
                self.reward = 10
                reward_given = True
            self.move_snakebody() # After the snake head moves, update the body.
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
            time.sleep(SLEEP)
            state = self.get_state()

    def step(self, action):
        '''
        moves the snake according to an agent's selected action
        '''
        if action == 0: self.go_up()
        if action == 1: self.go_right()
        if action == 2: self.go_down()
        if action == 3: self.go_left()
        self.run_game()
        return self.get_state(), self.reward, self.done

    def get_state(self):
        '''
        obtains the 12-dimensional state of the snake
        '''
        # Define the coordinates of the snake head in units of HEAD_SIZE.
        self.snake.x, self.snake.y = self.snake.xcor()/WIDTH, self.snake.ycor()/HEIGHT

        # Scale the coordinates of the snake head down to range from 0 to 1.
        # This requires shifting the origin to the left by half the width of the 1-length x-interval,
        # and also shifting down by half the height of the 1-length y-interval.
        self.snake.xsc, self.snake.ysc = self.snake.x/WIDTH+0.5, self.snake.y/HEIGHT+0.5

        # Scale the coordinates of the apple to range between 0 and 1 in the same way.
        self.apple.xsc, self.apple.ysc = self.apple.x/WIDTH+0.5, self.apple.y/HEIGHT+0.5

        # Check to see if a wall is next to the head and in what cardinal direction.
        wall_up=1 if self.snake.y >= HEIGHT/2 else 0
        wall_down=1 if self.snake.y <= -HEIGHT/2 else 0
        wall_right=1 if self.snake.x >= WIDTH/2 else 0
        wall_left=1 if self.snake.y <= -WIDTH/2 else 0

        # Check to see if the snake's body is in the head's immediate vicinity.
        # Here are some example states where ^ is the head and . is the tail:
        #
        # [o][1][o]     [.][1][o]       [^]
        # [1][^][1]        [^][1]       [o]
        # [.][1][o]        [1][o]       [.]
        body_up=body_down=body_right=body_left=False
        if len(self.snake_body) > 3: # [o][o]
                                     # [o][<] You need a length of at least 4 to eat yourself.
            for body in self.snake_body[3:]: # Only loop through the 3rd body link onward.
                if body.distance(self.snake) == HEAD_SIZE: # If the body is exactly one HEAD_SIZE unit away from the head
                    if body.ycor() < self.snake.ycor(): # if the body link is underneath the head
                        body_down = True # then confirm the link is below the head.
                    elif body.ycor() > self.snake.ycor(): # otherwise if the body link is above the head,
                        body_up = True # confirm the link is above the head.
                    if body.xcor() < self.snake.xcor():
                        body_left = True
                    elif body.xcor() > self.snake.xcor():
                        body_right = True

        # state:     apple_up,     apple_right,     apple_down,     apple_left,
        #         obstacle_up,  obstacle_right,  obstacle_down,  obstacle_left,
        #        direction_up, direction_right, direction_down, direction_left
        if self.state_definition_type == 'coordinates': # Let the agent receive direct knowledge of where the apple is and where the head is.
            state = [self.apple.xsc, self.apple.ysc, self.snake.xsc, self.snake.ysc, \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]

        elif self.state_definition_type == 'no direction': # Don't let the agent know which direction the snake is moving in.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    0, 0, 0, 0]

        elif self.state_definition_type == 'no body knowledge': # The agent no longer knows if the head is immediately near a body part, only walls.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    wall_up, wall_right, wall_down, wall_left, \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]

        else: # Allow the agent to know which cardinal directions the apple is in, whether the head is immediately next to walls or body links, and what direction the head is moving in.
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
        return state

if __name__ == '__main__':
    env = Snake(human=HUMAN)
    if HUMAN:
        while True:
            env.run_game()
