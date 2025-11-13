from env import *
from custom_CNN import *
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from callback import *
import os
import sys
import stable_baselines3


# Initialize wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env setting
def make_env():
    if env_name == 'frame_skip':
        return Monitor(FrameSkipSandTetrisEnv(SandTetrisEnv(), gamma=hyperparams['gamma'], skip=PPO_apply_dur, simulate_falling=simulate_falling))
    elif env_name == 'block2sand':
        return Monitor(Block2Sand_SandTetrisEnv(SandTetrisEnv(), gamma=hyperparams['gamma'], simulate_falling=simulate_falling))
env = DummyVecEnv([make_env for _ in range(num_cpu)])

# training
model_load_path = f"./model/ppo_sand_tetris_learn_{load_file_num}_{start_timesteps}"
if os.path.exists(model_load_path + ".zip"):
    run = wandb.init(project='sand_tetris_rl', name=f'ppo_{load_file_num}_{start_timesteps}')
    model = PPO.load(model_load_path, env=env, tensorboard_log=f"runs/{run.id}", device=device)
    print("Model loaded from", model_load_path)
    model.learn(total_timesteps=timesteps, callback=[WandbCallbackcustom(), SaveOnStepCallback(start_timesteps)])
else:
    name = f'ppo_{save_file_num}'
    for k in hyperparams.items():
        name += '_' + k[0]
        name += '_' + str(k[1]) 
    run = wandb.init(project='sand_tetris_rl', name=name)
    policy_kwargs = dict(
        features_extractor_class=SandTetrisCNN_V3,
        features_extractor_kwargs=dict(features_dim=128),
        log_std_init = log_std_init,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run.id}",
        **hyperparams    
    )
    model.learn(total_timesteps=timesteps, callback=[WandbCallbackcustom(), SaveOnStepCallback(0)])
run.finish()

# rendering
if render_after_training:
    action = 0
    dur_frame = 0
    window = pyglet.window.Window(screen_width, screen_height, "Sand Tetris")
    env = SandTetrisEnv()

    @window.event
    def on_draw():
        window.clear()
        env.render()

    @window.event
    def update(dt):
        global obs, dur_frame, action
        if not dur_frame % PPO_apply_dur:   action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action, dur_frame)
        if done:    obs, info = env.reset()
        dur_frame = (dur_frame + 1) % PPO_apply_dur

    pyglet.clock.schedule_interval(update, 1/60)

    obs, info = env.reset()
    pyglet.app.run()