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

def evaluate_one_level(cfg, env):
    log_rewards = dict()
    first_rewards = deque(maxlen=256) # first episode!
    final_rewards = deque(maxlen=50)
    model_path = join('/home/mandi/stable-baselines3/meta-rl/log', cfg.load_run)
    load_step = cfg.load_step
    saved_cfg = OmegaConf.load(join(model_path, 'config.yaml'))
    
    modify_env = True 
    if modify_env:
        saved_cfg.ppo_lstm.policy_kwargs.normalize_images = False
        env.observation_space = spaces.Dict(
            {'rgb': spaces.Box(low=-25.0, high=255.0, shape=(65, 64, 3))}
        )
    model = PPO(env=env, **saved_cfg.ppo_lstm)
    load_path = join(model_path, 'eval/models/{}'.format(load_step))
    assert os.path.exists(load_path+'.zip'), 'Model not found'
    model = model.load(env=env, path=load_path ) 

    print('loaded model')
    model.policy.set_training_mode(False)
     
    n_steps = 0
    n_envs = cfg.env.num_envs

    n_rollout_steps = int(cfg.total_steps)
    if model.use_sde:
        model.policy.reset_noise(n_envs)
    n_layer, _, hidden = model._last_lstm_states.pi[0].shape
    single_hidden_state_shape = (n_layer, n_envs, hidden)
    lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(model.device),
                th.zeros(single_hidden_state_shape).to(model.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(model.device),
                th.zeros(single_hidden_state_shape).to(model.device),
            ),
        )
    

    last_obs = env.reset()
    if modify_env:
        # assert cfg.ppo_lstm.policy_kwargs.normalize_images == False, 'Custom env assumes image already normalized'
        extras = np.zeros((n_envs, 1, 64, 3), dtype=np.float32)
        extras[:, 0, :] = [4, 0.0, 0.0]
        if last_obs['rgb'].shape[1] == 3:
            last_obs['rgb'] = np.transpose(last_obs['rgb'], (0, 2, 3, 1))
        # print(extras.shape)
        last_obs['rgb'] = np.concatenate([last_obs['rgb']/255.0, extras], axis=1)
        last_obs['rgb'] = np.transpose(last_obs['rgb'], (0, 3, 1, 2))
    dones = np.ones((n_envs,), dtype=bool) # model._last_episode_starts
    first_episode_dones = np.zeros((n_envs,), dtype=bool)
    while n_steps < n_rollout_steps:
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(last_obs, model.device) 
            episode_starts = th.tensor(dones).float().to(model.device)
            actions, values, log_probs, lstm_states = model.policy.forward(
                obs_tensor, lstm_states, episode_starts)

        actions = actions.cpu().numpy()
        clipped_actions = actions 
        if isinstance(model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)
        new_obs, rewards, dones, infos = env.step(clipped_actions)

        
        if modify_env:
            extras = np.zeros((n_envs, 1, 64, 3), dtype=np.float32)
             
            for idx, done_ in enumerate(dones):
                if done_:
                    if first_episode_dones[idx] == 0:
                        first_episode_dones[idx] = 1
                        first_rewards.append(rewards[idx])
                    # tot_eps[idx] += 1
                    # print(f'finished {tot_eps} episode, reward {current_reward}')
                    if infos[idx].get('episode', None) is not None:
                        final_rewards.append(infos[idx]['episode']['r'])
                    dones[idx] = False 
            tocat = np.stack([actions, rewards, dones.astype(np.float)], axis=1)
            tocat = tocat.reshape(n_envs, 1, 3)
            extras[:, 0, :] = tocat
            if new_obs['rgb'].shape[1] == 3:
                new_obs['rgb'] = np.transpose(new_obs['rgb'], (0, 2, 3, 1))
            new_obs['rgb'] = np.concatenate([new_obs['rgb']/255.0, extras], axis=1)
            new_obs['rgb'] = np.transpose(new_obs['rgb'], (0, 3, 1, 2))
        n_steps += n_envs
        if n_steps % cfg.log_freq == 0 or n_steps == n_rollout_steps - 1:
            log_rewards[int(n_steps/n_envs)] = np.mean(final_rewards) if len(final_rewards) > 0 else -1 

            # print(f'env stepped {n_steps/n_envs}', log_rewards[int(n_steps/n_envs)])
        for idx, done_ in enumerate(dones):
            #if rewards[idx] > 0:
            #    print('env idx {}, reward: {}'.format(idx, rewards[idx]))
            if done_: 
                print('all episodes are done within 1 trial', n_steps ) 
                #   print('env index {} done, step {}'.format(idx, n_steps), rewards[idx])
                lstm_states.pi[0][:, idx] = 0 
                lstm_states.pi[1][:, idx] = 0
                lstm_states.vf[0][:, idx] = 0
                lstm_states.vf[1][:, idx] = 0 
        last_obs = new_obs
    # need tne env to have only one level
    return log_rewards, final_rewards, np.mean(first_rewards) if len(first_rewards) > 0 else -1 


@hydra.main(config_name='eval_cfg', config_path='conf')
def main(cfg: DictConfig) -> None: 

    modify_env = cfg.use_custom 
    env_cfg = dict(cfg.env)  
    if modify_env:
        cfg.env.num_levels = 1 
    start_level = cfg.env.start_level
    if not cfg.unseen:
        print('eval on trained levels!')
        start_level = 0 
    all_levels = dict()
    for start in range(10):
        cfg.env.start_level = start_level + start 
        env = ProcgenEnv(**(cfg.env))

        env = VecMonitor(env) 
        env = VecTransposeImage(env)
        print('made env at starting level', start)
  
        log_rewards, final_rewards, first_rew = evaluate_one_level(cfg, env)
        all_levels[start] = (log_rewards, final_rewards, first_rew)

    log_steps = defaultdict(list)
    for level, logs in all_levels.items():
        log_rewards = logs[0]
        for step, reward in (log_rewards).items():
            log_steps[step].append(reward)
    
    log_means = dict()
    log_stds = dict()
    for step, means in log_steps.items():
        log_means[step] = np.mean(means)
        log_stds[step] = np.std(means)

    wandb_name = (cfg.load_run.split('/')[-2])
    if cfg.unseen:
        wandb_name = 'Unseen' + wandb_name
    else:
        wandb_name = 'Seen' + wandb_name
    run = wandb.init(
            name=wandb_name,
            config=dict(cfg),
            **cfg.wandb
            )
    for step in log_means.keys():
        wandb.log(
            {'eval env step': int(step), 'reward_mean': log_means[step], 'reward_std': log_stds[step]},
        )
 
    for level, logs in all_levels.items():
        wandb.log(
            {'eval task level': int(level), 'first episode reward': logs[2] }
        )
    print(log_means, log_stds)
    return 

    current_rewards = [] 
    
    

    model_path = join('/home/mandi/stable-baselines3/meta-rl/log', cfg.load_run)
    load_step = cfg.load_step
    saved_cfg = OmegaConf.load(join(model_path, 'config.yaml'))
    if modify_env:
        saved_cfg.ppo_lstm.policy_kwargs.normalize_images = False
    
    # assume recurrent model
    if modify_env:
        env.observation_space = spaces.Dict(
            {'rgb': spaces.Box(low=-25.0, high=255.0, shape=(65, 64, 3))}
        )
    model = PPO(env=env, **saved_cfg.ppo_lstm)
    load_path = join(model_path, 'eval/models/{}'.format(load_step))
    assert os.path.exists(load_path+'.zip'), 'Model not found'
    model = model.load(env=env, path=load_path ) 

    print('loaded model')
    model.policy.set_training_mode(False)
     
    n_steps = 0
    n_envs = cfg.env.num_envs
    log_rewards = deque(maxlen=50)
    n_rollout_steps = int(cfg.total_steps)
    if model.use_sde:
        model.policy.reset_noise(n_envs)
    n_layer, _, hidden = model._last_lstm_states.pi[0].shape
    single_hidden_state_shape = (n_layer, n_envs, hidden)
    lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(model.device),
                th.zeros(single_hidden_state_shape).to(model.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(model.device),
                th.zeros(single_hidden_state_shape).to(model.device),
            ),
        )
    

    last_obs = env.reset()
    if modify_env:
        # assert cfg.ppo_lstm.policy_kwargs.normalize_images == False, 'Custom env assumes image already normalized'
        extras = np.zeros((n_envs, 1, 64, 3), dtype=np.float32)
        extras[:, 0, :] = [4, 0.0, 0.0]
        if last_obs['rgb'].shape[1] == 3:
            last_obs['rgb'] = np.transpose(last_obs['rgb'], (0, 2, 3, 1))
        # print(extras.shape)
        last_obs['rgb'] = np.concatenate([last_obs['rgb']/255.0, extras], axis=1)
        last_obs['rgb'] = np.transpose(last_obs['rgb'], (0, 3, 1, 2))
    dones = np.ones((n_envs,), dtype=bool) # model._last_episode_starts

    while n_steps < n_rollout_steps:
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(last_obs, model.device) 
            episode_starts = th.tensor(dones).float().to(model.device)
            actions, values, log_probs, lstm_states = model.policy.forward(
                obs_tensor, lstm_states, episode_starts)

        actions = actions.cpu().numpy()
        clipped_actions = actions 
        if isinstance(model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)
        new_obs, rewards, dones, infos = env.step(clipped_actions)

        if modify_env:
            extras = np.zeros((n_envs, 1, 64, 3), dtype=np.float32)
             
            for idx, done_ in enumerate(dones):
                if done_:
                    # tot_eps[idx] += 1
                    # print(f'finished {tot_eps} episode, reward {current_reward}')
                    if infos[idx].get('episode', None) is not None:
                        log_rewards.append(infos[idx]['episode']['r'])
                    dones[idx] = False 
            tocat = np.stack([actions, rewards, dones.astype(np.float)], axis=1)
            tocat = tocat.reshape(n_envs, 1, 3)
            extras[:, 0, :] = tocat
            if new_obs['rgb'].shape[1] == 3:
                new_obs['rgb'] = np.transpose(new_obs['rgb'], (0, 2, 3, 1))
            new_obs['rgb'] = np.concatenate([new_obs['rgb']/255.0, extras], axis=1)
            new_obs['rgb'] = np.transpose(new_obs['rgb'], (0, 3, 1, 2))
        n_steps += n_envs
        if len(log_rewards) > 0:
            print(f'env stepped {n_steps}', np.mean(log_rewards))
        for idx, done_ in enumerate(dones):
            #if rewards[idx] > 0:
            #    print('env idx {}, reward: {}'.format(idx, rewards[idx]))
            if done_: 
                print('all episodes are done within 1 trial', n_steps )

                #   print('env index {} done, step {}'.format(idx, n_steps), rewards[idx])
                lstm_states.pi[0][:, idx] = 0 
                lstm_states.pi[1][:, idx] = 0
                lstm_states.vf[0][:, idx] = 0
                lstm_states.vf[1][:, idx] = 0
            # if (
            #     done_
            #     and infos[idx].get("terminal_observation") is not None
            #     and infos[idx].get("TimeLimit.truncated", False)
            #     ):
               # print(infos[idx])
        last_obs = new_obs
    # print('total eps: {}, succ eps: {}'.format(tot_eps, succ_ep), succ_ep/tot_eps)



# env = ProcgenEnv(
#     env_name='coinrun', 
#     num_levels=1, 
#     start_level=10000, 
#     num_envs=1, 
#     restrict_themes=False, 
#     distribution_mode='hard',
#     render_mode="rgb_array",
#     )
# env.reset()
# # print(env.action_space.sample()): gives 1 number, need to expand to array 
# for i in range(1020):
#     ob, rew, first, info = env.step(np.ones(2) )
#     if True in first:
#         print(i, first)

# print(first, info)
# print(env.env.observe())

# print(env.env.get_state()[0])
# imgs = np.concatenate(env.reset()['rgb'], axis=0)
# ls = []
# for i in range(10):
#     env = ProcgenEnv(
#     env_name='coinrun', 
#     num_levels=1, 
#     start_level=10010+i, 
#     num_envs=1, 
#     restrict_themes=False, 
#     distribution_mode='hard',
#     render_mode="rgb_array",
#     )
#     ls.append(env.reset()['rgb'][0])
#     env.close()

# imgs = np.concatenate(ls, axis=1)

# plt.imsave('harder10.png', imgs)

# env = ProcgenEnv(
#     env_name='coinrun', 
#     num_levels=10, 
#     start_level=10000, 
#     num_envs=10, 
#     restrict_themes=False, 
#     distribution_mode='hard'
#     )
# imgs = np.concatenate(env.reset()['rgb'], axis=0)
# plt.imsave('ob1_another10.png', imgs)

if __name__ == '__main__':
    main()