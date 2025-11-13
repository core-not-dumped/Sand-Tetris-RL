from variables.color import *
import numpy as np

screen_width = 540
screen_height = 960

# idx: particle map, coor: window coor
def particle_idx_to_coor(x, y):
    return field_left + x * particle_size, field_bottom + y * particle_size

def particle_coor_to_idx(x, y):
    return (x - field_left) // particle_size, (y - field_bottom) // particle_size


# field
field_thickness = 2
# set field_left % particle_size == 0
field_left = 42
# set filed_right % particle_size == 0
field_right = 399
field_width = field_right - field_left 
# set field_top % particle_size == 0
field_top = 800
# set field_bottom % particle_size == 0
field_bottom = 100
field_height = field_top - field_bottom


# block
remove_prob = 0.0
block_size = 28
block_start_x = (field_left + field_right) // 2 - block_size
block_start_y = field_top - block_size
# actual speed -> block_falling_spd * particle_size
block_falling_spd = -1
block_x_spd = 1
block_color_list = [GREEN, RED, BLUE, YELLOW]
num_type = len(block_color_list)
channel_num = len(block_color_list) + 1 # if num_type + 1 == channel_num -> last channel이 falling block check channel임
# first (0, 0) is standard of the rotation
block_coor_list = [
    [[[0, 0], [-1, 0], [1, 0], [2, 0]], [[0, 0], [0, 1], [0, -1], [0, -2]]],\
    [[[0, 0], [-1, 0], [1, 0], [0, -1]], [[0, 0], [0, 1], [0, -1], [1, 0]], [[0, 0], [-1, 0], [0, 1], [1, 0]], [[0, 0], [-1, 0], [0, 1], [0, -1]]],\
    [[[0, 0], [1, 0], [0, -1], [1, -1]]],\
    [[[0, 0], [-1, 0], [1, 0], [-1, -1]], [[0, 0], [0, 1], [0, -1], [1, -1]], [[0, 0], [-1, 0], [1, 0], [1, 1]], [[0, 0], [-1, 1], [0, 1], [0, -1]]],\
    [[[0, 0], [-1, 0], [1, 0], [1, -1]], [[0, 0], [0, 1], [1, 1], [0, -1]], [[0, 0], [-1, 1], [-1, 0], [1, 0]], [[0, 0], [-1, 1], [0, 1], [0, -1]]],\
    [[[0, 0], [-1, 0], [0, -1], [1, -1]], [[0, 0], [0, -1], [1, 0], [1, 1]]],\
    [[[0, 0], [-1, -1], [0, -1], [1, 0]], [[0, 0], [0, 1], [1, 0], [1, -1]]],\
]


# particle
particle_size = 7
particle_map_width = field_width // particle_size
particle_map_height = field_height // particle_size + block_size // particle_size
falling_freq = 2


# reinforcemnet learning
num_cpu = 12
def linear_schedule_min(initial_value, min_ratio=1/20):
    def func(progress_remaining: float):
        return initial_value * (min_ratio + (1 - min_ratio) * progress_remaining)
    return func
hyperparams = {
    'learning_rate': np.random.choice([linear_schedule_min(2e-4, min_ratio=1/20)]),
    'gamma': np.random.choice([0.98]),
    'n_steps': np.random.choice([2048]),
    'clip_range': np.random.choice([0.25]),
    'gae_lambda': np.random.choice([0.95]),
    'ent_coef': np.random.choice([0.002]),
    'vf_coef': np.random.choice([0.05]),
    'batch_size': np.random.choice([256]),
    'n_epochs': np.random.choice([4]),
    'max_grad_norm': np.random.choice([0.5])
}
env_name = 'frame_skip'#['frame_skip', 'block2sand']
action_num = {'frame_skip':3, 'block2sand':1}[env_name]
time_reward = 0
use_block_reward = False
block_reward = 0.002
line_reward = 0.7
done_reward = -2.0
PPO_apply_dur = 4
height_reward_max = {'frame_skip':1.0, 'block2sand':0.5}[env_name]
BFS_freq = PPO_apply_dur
wait_next_block_frame = 0
simulate_falling = 20
misaligned_reward = 0.0
log_std_init = -1
finish_line_num = 2.5

save_file_num = 1
load_file_num = 5

episode_reward_mean_num = 100
model_save_freq = 1000000 // num_cpu
start_timesteps = 123123
timesteps = 20000000

env_std = 1/2
env_mean = 1/2
#env_std = 0.166
#env_mean = 0.315

render_after_training = False
