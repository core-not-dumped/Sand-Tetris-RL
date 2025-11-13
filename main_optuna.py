import os
import optuna
import torch
import wandb
import numpy as np
from env import *
from custom_CNN import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from callback import *
import json

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(gamma):
    return Monitor(FrameSkipSandTetrisEnv(SandTetrisEnv(), gamma=gamma, skip=PPO_apply_dur, simulate_falling=simulate_falling))

def optimize_ppo(trial):
    # Optuna로 탐색할 하이퍼파라미터 범위 정의
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 3e-5, 5e-4, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.98),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.05),
        'vf_coef': trial.suggest_float('vf_coef', 0.3, 1.0),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'n_epochs': trial.suggest_categorical('n_epochs', [5, 10, 15]),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0)
    }

    env = DummyVecEnv([lambda: make_env(hyperparams['gamma']) for _ in range(num_cpu)])

    # wandb 실행 초기화
    name = f'ppo_{save_file_num}'
    for k in hyperparams.items():
        name += '_' + k[0]
        name += '_' + str(k[1]) 
    run = wandb.init(project='sand_tetris_rl', name=name)

    # policy 설정
    policy_kwargs = dict(
        features_extractor_class=SandTetrisCNN_V3,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    # PPO 모델 생성
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        device=device,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        **hyperparams
    )

    # 학습 및 콜백
    callback = WandbCallbackcustom()
    model.learn(total_timesteps=1000000, callback=callback)

    # 평가: 마지막 rollout의 평균 reward 사용
    mean_reward = np.mean(callback.recent_rewards) if hasattr(callback, "recent_rewards") else 0.0

    run.finish()
    env.close()
    return mean_reward


if __name__ == "__main__":
    # Optuna study 설정
    study = optuna.create_study(direction="maximize", study_name="ppo_sand_tetris_optuna")
    study.optimize(optimize_ppo, n_trials=50, n_jobs=1)

    print("🎯 Best trial:")
    print(study.best_trial.params)

    # JSON 파일로 저장
    os.makedirs("optuna_results", exist_ok=True)
    best_params_path = "optuna_results/best_hyperparams.json"

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=4, ensure_ascii=False)

    print(f"✅ Best hyperparameters saved to {best_params_path}")