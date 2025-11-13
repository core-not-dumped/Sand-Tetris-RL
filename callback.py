import wandb
import os
from stable_baselines3.common.callbacks import BaseCallback
from variables.global_variables import *
from collections import deque


class WandbCallbackcustom(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallbackcustom, self).__init__(verbose)
        self.recent_rewards = deque(maxlen=episode_reward_mean_num)  # Track the rewards of the last 30 episodes
        self.current_rewards = [0] * num_cpu   # Track current rewards for each environment
        self.recent_lengths = deque(maxlen=episode_reward_mean_num)
        self.current_lengths = [0] * num_cpu

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0] * num_cpu)
        dones = self.locals['dones']

        for i in range(num_cpu):
            self.current_rewards[i] += rewards[i]
            self.current_lengths[i] += 1

            if dones[i]:
                self.recent_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0  # Reset current reward for the next episode
                self.recent_lengths.append(self.current_lengths[i])
                self.current_lengths[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        if len(self.recent_rewards) > 0:
            average_recent_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            wandb.log({'episode_mean_reward': average_recent_reward})
        if len(self.recent_lengths) > 0:
            average_episode_length = sum(self.recent_lengths) / len(self.recent_lengths)
            wandb.log({'episode_mean_length': average_episode_length})
        
        wandb.log({
            'approx_kl': self.model.logger.name_to_value.get('train/approx_kl', 0),
            'value_loss': self.model.logger.name_to_value.get('train/value_loss', 0),
            'entropy_loss': self.model.logger.name_to_value.get('train/entropy_loss', 0),
            'explained_variance': self.model.logger.name_to_value.get('train/explained_variance', 0),
            'clip_fraction': self.model.logger.name_to_value.get('train/clip_fraction', 0),
            'policy_gradient_loss': self.model.logger.name_to_value.get('train/policy_gradient_loss', 0),
        })

class SaveOnStepCallback(BaseCallback):
    def __init__(self, start_time: int, verbose: int = 1):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.start_time = start_time

    def _on_step(self) -> bool:
        if self.n_calls % model_save_freq == 0:
            model_path = f'./model/ppo_sand_tetris_learn_{save_file_num}_{self.start_time + self.n_calls}'
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at {self.n_calls} steps to {model_path}")
        return True