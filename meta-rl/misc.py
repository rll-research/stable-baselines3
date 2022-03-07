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

def main():
    n_envs = 1
    modify_env = True 
    max_trials = 5
    cfg = OmegaConf.load('/home/mandi/stable-baselines3/meta-rl/conf/config.yaml')
    
    env_cfg = dict(cfg.custom_env.train) # or cfg.env.train.
    env_cfg['num_levels'] = 50
    if modify_env:
        env_cfg['num_levels'] = 1 
        env_cfg['start_level'] = 100010
    env_cfg['render_mode'] ='rgb_array'
    # env = SubprocVecEnv([lambda : make_custom_env(
    #     env_cfg) for i in range(n_envs)])
    # env = DummyVecEnv([lambda : make_custom_env(env_cfg) for i in range(n_envs)])
    # env = VecMonitor(env) 
    # frame = env.render('rgb_array')
    # print(frame.shape)
    # return 
    cfg.env.train.num_envs = n_envs 
    if modify_env:
        cfg.env.train.num_levels = 1 
        cfg.env.train.start_level = 500 #100020
    env = ProcgenEnv(**(cfg.env.train), render_mode='rgb_array')
    
    current_reward = 0 
    
    env = VecMonitor(env) 
    print('made env')

    model_path = 'log/LSTM-Nstep256-NewSeqProcess-NoEpisodeStart-ResetStates/seed1/'
    load_step = 1300 

    model_path = 'log/LSTM-1x256-NewProcessSeq-PerEnvSample-Impala-CustonEnv1e5/seed1'
    load_step = 1500 
    cfg = OmegaConf.load(join(model_path, 'config.yaml'))
    if modify_env:
        cfg.ppo_lstm.policy_kwargs.normalize_images = False
    
    # assume recurrent model
    if modify_env:
        env.observation_space = spaces.Dict(
            {'rgb': spaces.Box(low=-25.0, high=255.0, shape=(65, 64, 3))}
        )
    model = PPO(env=env, **cfg.ppo_lstm)
    load_path = join(model_path, 'eval/models/{}'.format(load_step))
    assert os.path.exists(load_path+'.zip'), 'Model not found'
    model = model.load(env=env, path=load_path ) 
    model.n_envs = n_envs
    print('loaded model')
    model.policy.set_training_mode(False)
     
    n_steps = 0
    tot_eps, succ_ep = 0, 0
    n_rollout_steps = 1000 
    if model.use_sde:
        model.policy.reset_noise(env.num_envs)
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
    env = VecTransposeImage(env)
    env = VecEvalVideoRecorder(
        env, video_folder='/home/mandi/stable-baselines3/meta-rl/videos')
    last_obs = env.reset()
    if modify_env:
        assert n_envs == 1
        # assert cfg.ppo_lstm.policy_kwargs.normalize_images == False, 'Custom env assumes image already normalized'
        extras = np.zeros((1, 1, 64, 3), dtype=np.float32)
        extras[0, 0, :] = [4, 0.0, 0.0]
        # print(last_obs['rgb'].shape, np.transpose(last_obs['rgb'], (0, 2, 3, 1)).shape)
        # print(extras.shape)
        last_obs['rgb'] = np.concatenate([last_obs['rgb']/255.0, extras], axis=1)
        last_obs['rgb'] = np.transpose(last_obs['rgb'], (0, 3, 1, 2))
    dones = np.ones((env.num_envs,), dtype=bool) # model._last_episode_starts

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
            extras = np.zeros((1, 1, 64, 3), dtype=np.float32)
            if rewards[0] > 0:
                current_reward += rewards[0]
            if dones[0]: 
                tot_eps += 1
                print(f'finished {tot_eps} episode, reward {current_reward}')
                current_reward = 0
                if tot_eps < max_trials:
                    dones[0] = False 
            
            extras[0, 0, :] = [clipped_actions[0], rewards[0], float(dones[0])]
            new_obs['rgb'] = np.concatenate([new_obs['rgb']/255.0, extras], axis=1)
            new_obs['rgb'] = np.transpose(new_obs['rgb'], (0, 3, 1, 2))
        n_steps += 1
        for idx, done_ in enumerate(dones):
            #if rewards[idx] > 0:
            #    print('env idx {}, reward: {}'.format(idx, rewards[idx]))
            if done_: 
                print('all episodes are done within 1 trial', n_steps )
                current_reward = 0
                tot_eps = 0
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