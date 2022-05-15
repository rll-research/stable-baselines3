# itr1: rew 0.3
# completely random actions: 8.1
# itr 4
import os
import gym
import numpy as np
import logging
import torch as th 
from collections import deque
from os.path import join
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
import hydra 
from omegaconf import DictConfig, OmegaConf, ListConfig
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    VecMonitor
)
from collections import defaultdict
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure as configure_logger
import wandb 
from custom.callbacks import LogEvalCallback
from custom.custom_procgen_env import MultiProcGenEnv, make_custom_env
import numpy as np
from procgen import ProcgenEnv
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from os.path import join
from gym import spaces 
from custom.custom_procgen_env import MultiProcGenEnv, make_custom_env
from custom.vec_eval_video_recorder import VecEvalVideoRecorder
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import torch as th 
import torch 
from copy import deepcopy 
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
import gym
import os 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import RNNStates

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None: 
    run = 'Scratch-Unseen10016-Seed123-256Envs-2048Batch/seed1'
    step = 4
    toload = join('/home/mandi/stable-baselines3/meta-rl/log', run)
    toload = join(toload, f'eval/models/{step}')
    toload2 = join('/home/mandi/stable-baselines3/meta-rl/log', run, 'eval/models/2')
    

    n_envs = 2
    cfg.procgen.eval.num_envs = n_envs
    cfg.procgen.eval.num_levels = 1
    cfg.procgen.eval.start_level = 10009 # 10016 
    env = ProcgenEnv(**(cfg.procgen.eval))
    env = VecMonitor(env)
    print('Made env')

    model = PPO(env=env, **cfg.ppo)
    model = model.load(env=env, path=toload) 
    # check if parameters are the same between model and model2:
    # for p1, p2 in zip(model.policy.mlp_extractor.parameters(), model2.policy.mlp_extractor.parameters()):
    #     print(p1.data.eq(p2.data).all())
    # raise ValueError
 

    model.policy.set_training_mode(False)
    n_steps = 0
    

    if model.use_sde:
        model.policy.reset_noise(n_envs)

    env = model.env
    last_obs = env.reset()
    states = None 
    dones = np.ones((n_envs,), dtype=bool) # model._last_episode_starts
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    first_episode_dones = np.zeros((n_envs,), dtype=bool)
    eps_rews = []
    while len(eps_rews) < 100:
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            # obs_tensor = obs_as_tensor(last_obs, model.device) 
            # obs_tensor = obs_tensor.transpose((0, 3, 1, 2)) 


            # actions, states = model.predict(last_obs, state=states, episode_start=episode_starts, 
            # deterministic=True)
            # obs_tensor = obs_as_tensor(last_obs, model.device)
            # forward_act, values, log_probs = model.policy(obs_tensor)
            # #print(actions, forward_act)
            # actions = forward_act.cpu().numpy()

            actions = np.random.randint(0, env.action_space.n, size=(n_envs,))

        #actions = actions.cpu().numpy()
        clipped_actions = actions 
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        # print(actions)

        for idx, done_ in enumerate(dones):
            #if rewards[idx] > 0:
            #    print('env idx {}, reward: {}'.format(idx, rewards[idx]))
            episode_starts[idx] = done_
            if done_: 
                print('env index {} done, step {}'.format(idx, n_steps), rewards[idx]) #, infos)
                eps_rews.append(rewards[idx])
        last_obs = new_obs
    print("Done", np.mean(eps_rews))
if __name__ == '__main__':
    main()