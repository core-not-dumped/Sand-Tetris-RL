import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import *
from custom_CNN import *
from stable_baselines3 import PPO
from callback import *

action = 0
dur_frame = 0
load_model_name = 'case2_25M'
window = pyglet.window.Window(screen_width, screen_height, "Sand Tetris")
env = SandTetrisEnv()
model = PPO.load(f"./model/{load_model_name}", env=env)

@window.event
def on_draw():
    window.clear()
    env.render()

@window.event
def update(dt):
    global obs, dur_frame, action
    if not dur_frame % PPO_apply_dur:   action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action, dur_frame)
    if done:    obs, info = env.reset()
    dur_frame = (dur_frame + 1) % PPO_apply_dur

pyglet.clock.schedule_interval(update, 1/60)

obs, info = env.reset()
pyglet.app.run()