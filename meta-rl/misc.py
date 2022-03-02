import numpy as np
from procgen import ProcgenEnv
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from os.path import join
from custom.custom_procgen_env import MultiProcGenEnv, make_custom_env
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import torch as th 
import torch 
from copy import deepcopy 
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
import gym
import os 
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    n_envs = 3
    cfg = OmegaConf.load('/home/mandi/stable-baselines3/meta-rl/conf/config.yaml')
    cfg.custom_env.train.num_levels = 2
    # env = SubprocVecEnv([lambda : make_custom_env(cfg.custom_env.train) for i in range(n_envs)])

    cfg.env.train.num_envs = n_envs
    env = ProcgenEnv(**(cfg.env.train))
    
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    print('made env')

    model_path = 'log/LSTM-Nstep256-NewSeqProcess-NoEpisodeStart-ResetStates/seed1/'
    load_step = 1300 
    # assume recurrent model
    model = PPO(env=env, **cfg.ppo_lstm)
    load_path = join(model_path, 'eval/models/{}'.format(load_step))
    assert os.path.exists(load_path+'.zip'), 'Model not found'
    model = model.load(env=env, path=load_path ) 
    model.n_envs = n_envs
    model._setup_model()
    print('loaded model')
    model.policy.set_training_mode(False)
    print(
        evaluate_policy(model, env, n_eval_episodes=100)
    )
    raise ValueError


    n_steps = 0
    tot_eps, succ_ep = 0, 0
    n_rollout_steps = 6000 
    if model.use_sde:
        model.policy.reset_noise(env.num_envs)
    lstm_states = deepcopy(model._last_lstm_states)
    last_obs = env.reset()
    dones = model._last_episode_starts
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
        n_steps += 1
        for idx, done_ in enumerate(dones):
            if infos[idx]['trial_num'] > 0:
                print(infos[idx]['trial_num'])
            if rewards[idx] > 0:
                print('env idx {}, reward: {}'.format(idx, rewards[idx]))
            if done_:
                
                tot_eps += 1
                if rewards[idx] > 0:
                    succ_ep += 1
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
    print('total eps: {}, succ eps: {}'.format(tot_eps, succ_ep), succ_ep/tot_eps)



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