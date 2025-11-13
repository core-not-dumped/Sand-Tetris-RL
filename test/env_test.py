import pyglet
from pyglet.gl import *
from pyglet import shapes
from pyglet.window import key

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from variables.global_variables import *

from draw.field import *
from objects.block import *
from objects.particles import *

from env import *

frameskiptest = True
skip_frame = 4
env_name = 'frame_skip'

window = pyglet.window.Window(screen_width, screen_height, "Sand Tetris")
if frameskiptest:
    if env_name == 'frame_skip':
        env = FrameSkipSandTetrisEnv(SandTetrisEnv(), hyperparams['gamma'], skip=skip_frame, simulate_falling=simulate_falling)
    elif env_name == 'block2sand':
        env = Block2Sand_SandTetrisEnv(SandTetrisEnv(), hyperparams['gamma'], simulate_falling=simulate_falling)
else:               env = SandTetrisEnv()
action = 0

def game_init():
    global field, blocks, particles

    field = Field()
    blocks = Blocks()
    particles = Particles()

game_init()

@window.event
def on_draw() -> None:
    window.clear()
    env.render()

@window.event
def update(dt):
    global obs, action
    if frameskiptest:
        if env_name == 'frame_skip':
            obs, rewards, done, truncated, info = env.step(action)
        elif env_name == 'block2sand':
            obs, rewards, done, truncated, info = env.step(action)
    else:
        obs, rewards, done, truncated, info = env.step(action, 0)
    if done:    obs, info = env.reset()


key_pressed = {'left':False, 'right':False, 'up':False, 'down':False}
@window.event
def on_key_press(symbol, modifier):
    global key_pressed, action
    if not frameskiptest or env_name == 'frame_skip':
        if symbol == key.LEFT:      action = 1
        if symbol == key.RIGHT:     action = 2
        if symbol == key.SPACE:     action = 3
    elif env_name == 'block2sand':
        if symbol == key.LEFT:      action = -1
        if symbol == key.RIGHT:     action = 1
        if symbol == key.SPACE:     action = 0

@window.event
def on_key_release(symbol, modifier):
    global key_pressed, action
    if symbol == key.LEFT:      action = 0
    if symbol == key.RIGHT:     action = 0
    if symbol == key.SPACE:     action = 0

pyglet.clock.schedule_interval(update, 1/50*(skip_frame if frameskiptest else 1))
pyglet.app.run()