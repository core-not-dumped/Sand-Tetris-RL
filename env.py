import gymnasium as gym
import pyglet
from variables.global_variables import *
import numpy as np
from draw.field import *
from draw.score import *
from objects.block import *
from objects.particles import *
import torch
from collections import deque
import torch.nn as nn
import torch.optim as optim

class RNDIntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env, lr=1e-4):
        super().__init__(env)

        # target network: (학습 X)
        # predictor network: 학습 O
        #self.target = RNDModel().eval()
        #self.predictor = RNDModel()

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        self.mse = nn.MSELoss(reduction='none')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # target & predictor output
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():   target_out = self.target(obs_tensor)
        pred_out = self.predictor(obs_tensor)

        # RND loss
        loss = self.mse(pred_out, target_out).mean()

        # predictor update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        intrinsic_reward = loss.item()
        reward += 0.01 * intrinsic_reward
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameSkipSandTetrisEnv(gym.Wrapper):
    def __init__(self, env, gamma, skip=4, simulate_falling=10):
        """Return only every `skip`-th frame."""
        super(FrameSkipSandTetrisEnv, self).__init__(env)
        self.simulate_falling = simulate_falling
        self.before_state_reward = 0
        self._skip = skip
        self.gamma = gamma

    def step(self, action):
        """Repeat action, sum reward, and return the last observation."""
        done = False
        truncated = False
        
        total_reward = 0.0
        for dur_frame in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action, dur_frame)
            total_reward += reward
            # if block collision and turn into particles
            if done or truncated:
                # add block reward
                total_reward += done_reward
                break

            if info['reward_give']:
                BFS_reward, state_reward = \
                    self.env.simulate_future_BFS(self.simulate_falling)
                total_reward += self.gamma * state_reward - self.before_state_reward
                self.before_state_reward = state_reward
                total_reward += BFS_reward
        
        # other rewards
        total_reward += time_reward
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self.before_state_reward = 0
        return self.env.reset(**kwargs)


class Block2Sand_SandTetrisEnv(gym.Wrapper):
    def __init__(self, env, gamma, simulate_falling=10):
        super(Block2Sand_SandTetrisEnv, self).__init__(env)
        self.simulate_falling = simulate_falling
        self.gamma = gamma

    def step(self, action):
        obs, reward, done, truncated, info = self.env.fast_step(action, self.gamma, self.simulate_falling)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class SandTetrisEnv(gym.Env):
    def __init__(self):
        super(SandTetrisEnv, self).__init__()

        self.key_pressed = {'left':False, 'right':False, 'up':False, 'down':False}
        
        self.field = Field()
        self.score = Score()
        self.blocks = Blocks()
        self.particles = Particles()
        
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(channel_num, particle_map_width, particle_map_height), dtype=np.float32)
        if env_name == 'block2sand':
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif env_name == 'frame_skip':
            self.action_space = gym.spaces.Discrete(action_num)
        self.before_state_reward = 0 # only use in block2sand env

        self.done = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def reset(self, seed=None):
        self.field = Field()
        self.blocks = Blocks()
        self.particles = Particles()
        self.key_pressed = {'left':False, 'right':False, 'up':False, 'down':False}
        self.done = False
        self.before_state_reward = 0 # only use in block2sand env
        return self.particles.state, {}


    def step(self, action, dur_frame):
        self.key_pressed = {'left':False, 'right':False, 'up':False, 'down':False}
        if action == 1:     self.key_pressed['left'] = True
        elif action == 2:   self.key_pressed['right'] = True
        elif action == 3 and dur_frame == 0:   self.blocks.rotate_blocks()

        total_reward = 0
        block_collision = False
        self.done = self.blocks.update(self.key_pressed, self.particles)
        block_collision = self.blocks.block_collision(self.particles)
        if block_collision: self.done = self.particles.return_top_height() > (particle_map_height - (block_size // particle_size) * finish_line_num)
        self.particles.update()
        self.particles.state_add_block(self.blocks)

        return self.particles.state, total_reward, self.done, False, {'reward_give':block_collision}

    
    def fast_step(self, action, gamma, simulate_falling):
        total_reward = 0
        block_final_pos = int(field_left + (field_right - field_left) * ((action + 1) / 2))
        while True:
            pres_pos = self.blocks.get_block_x_pos()
            self.key_pressed = {'left':False, 'right':False, 'up':False, 'down':False}
            if block_final_pos < pres_pos:     self.key_pressed['left'] = True
            elif block_final_pos > pres_pos:   self.key_pressed['right'] = True

            block_collision = False
            self.blocks.update(self.key_pressed, self.particles)
            block_collision = self.blocks.block_collision(self.particles)
            self.particles.update()
            if block_collision:
                self.done = self.particles.return_top_height() > (particle_map_height - (block_size // particle_size) * 4)
                BFS_reward, state_reward = self.simulate_future_BFS(simulate_falling)
                total_reward += gamma * state_reward - self.before_state_reward
                self.before_state_reward = state_reward
                total_reward += BFS_reward
                self.blocks.update(self.key_pressed, self.particles) # regenerate next block
                if self.done:   total_reward += done_reward
                break
        
        self.particles.state_add_block(self.blocks)
        return self.particles.state, total_reward, self.done, False, {}


    def render(self, mode='human'):
        self.field.draw()
        self.blocks.draw()
        self.particles.draw()
        self.score.draw(self.particles.total_score)

    def simulate_future_BFS(self, simulate_falling):
        return self.particles.future_falling_simulate(simulate_falling)