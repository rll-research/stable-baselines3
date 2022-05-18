"""

python train.py run_name=Hard-10-NatureCNN-AllTheme \
    procgen.train.num_levels=10 learn.total_timesteps=5e6 procgen.eval.num_levels=10 policy_cfg=DictObs
"""
import os
import gym
import numpy as np
import logging
import torch as th 
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
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure as configure_logger
import wandb 
from custom.callbacks import LogEvalCallback, NormalizeBufferRewardCallback
from custom.custom_procgen_env import MultiProcGenEnv, make_custom_env
from custom.mt_atari import make_multitask_atari_env
from copy import deepcopy
from natsort import natsorted
from glob import glob

def make_wandb_config(cfg):
    wandb_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            for k, v in val.items():
                if k == 'train' or k == 'eval':
                    for kk, vv in v.items():
                        wandb_cfg[f'{key}/{k}/{kk}'] = vv 
                else:
                    wandb_cfg[f'{key}/{k}'] = v
        else:
            wandb_cfg[key] = val  
    return wandb_cfg 
    
def evaluate(cfg):
    #assert cfg.env_type == 'procgen' and not cfg.use_custom, 'Only original procgen env is supported for now'

    logging.info(f"Eval **one at a time** on levels {cfg.env.eval.start_level} - {cfg.env.eval.start_level + cfg.env.eval.num_levels}")
    
    toload = join(cfg.data_path, cfg.load_run)
    steps = natsorted(glob(join(toload, 'eval/models/*.zip')))
    if len(steps) == 0:
        print('No model found in !', toload)
        return 
    logging.info(f"Load run {cfg.load_run}, found {len(steps)} checkpoints for models")
    print([step.split('/')[-1] for step in steps])


    start = cfg.env.eval.start_level
    total_levels = cfg.env.eval.num_levels
    env_cfg = deepcopy(cfg.env.eval)
    env_cfg.num_levels = 1
    n_envs = env_cfg.num_envs

    if cfg.log_wb:
        cfg.wandb.job_type = 'eval'
        cfg.log_path = ''
        cfg.learn.eval_log_path = ''
        wandb_cfg = make_wandb_config(cfg)
        run = wandb.init(
            name=cfg.run_name, 
            config=wandb_cfg,
            **cfg.wandb
            )
    for step in steps:
        toload = step[:-4] # remove .zip 
        model_itr = int(toload.split('/')[-1])
        level_data = dict()
        for i in range(total_levels):
            level = start + i
            env_cfg.start_level = level
            env = ProcgenEnv(**(env_cfg))
            env = VecMonitor(env)
            logging.info('Made env for level {}'.format(start + i))

            ppo_model = PPO(env=env, **cfg.ppo)
            model = ppo_model.load(env=env, path=toload) 
            model.policy.set_training_mode(False) 
            env = model.env
            last_obs = env.reset()

            states = None 
            dones = np.ones((n_envs,), dtype=bool) # model._last_episode_starts
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            first_episode_dones = np.zeros((n_envs,), dtype=bool)
            eps_rews = []
            while len(eps_rews) < 100:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(last_obs, model.device)
                    forward_act, values, log_probs = model.policy(obs_tensor)
                    clipped_actions = forward_act.cpu().numpy() # no clip for procgen
                
                new_obs, rewards, dones, infos = env.step(clipped_actions)
                for idx, done_ in enumerate(dones):
                    episode_starts[idx] = done_
                    if done_: 
                        #print('env index {} done, step {}'.format(idx, n_steps), rewards[idx]) #, infos)
                        eps_rews.append(rewards[idx])
                last_obs = new_obs
            
            mean_rew = np.mean(eps_rews)
            std_rew = np.std(eps_rews)
            logging.info(f"Level {level}, model itr {model_itr}, mean reward {mean_rew}, std {std_rew}")
            level_data[f"level{level}/reward_mean"] = mean_rew
            level_data[f"level{level}/reward_std"] = std_rew

        level_data['model_itr'] = model_itr
        if cfg.log_wb:
            wandb.log(level_data)
    return

def evaluate_rl2(cfg):
    toload = join(cfg.data_path, cfg.load_run)
    steps = natsorted(glob(join(toload, 'eval/models/*.zip')))
    if len(steps) == 0:
        print('No model found in !', toload)
        return 
    logging.info(f"Load run {cfg.load_run}, found {len(steps)} checkpoints for models, use the last model")
    print([step.split('/')[-1] for step in steps])
    step = steps[-2]

    cfg.env = cfg.procgen_custom 
    env_cfg = deepcopy(cfg.env.eval) 
    total_levels = cfg.env.eval.num_levels
    start = cfg.env.eval.start_level
    env_cfg.num_levels = 1
    
    logging.info('Using custom procgen env! Max number of trials: %d' % cfg.procgen_custom.train.max_trials)
    n_envs = cfg.env.num_eval_env
    print('Making envs parallel:', n_envs)

    if cfg.log_wb:
        cfg.wandb.job_type = 'eval'
        cfg.log_path = ''
        cfg.learn.eval_log_path = ''
        wandb_cfg = make_wandb_config(cfg)
        run = wandb.init(
            name=cfg.run_name, 
            config=wandb_cfg,
            **cfg.wandb
            )
    
    toload = step[:-4] # remove .zip 
    model_itr = int(toload.split('/')[-1])
    level_data = dict()
    for i in range(total_levels):
        level = start + i
        env_cfg.start_level = level
        env = SubprocVecEnv([
            lambda : make_custom_env(env_cfg) for i in range(n_envs)])
        env = VecMonitor(env, info_keywords=["past_rewards"])
        
        ppo_model = PPO(env=env, **cfg.ppo)
        model = ppo_model.load(env=env, path=toload) 
        model.policy.set_training_mode(False) 
        env = model.env
        print('Model envs',model.n_envs, env.num_envs)
        #raise ValueError

        last_obs = env.reset()
        dones = np.ones((n_envs,), dtype=bool) # model.
        _last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        env_steps = 0 
        past_rew = []
        final_rs, final_ts = [], [] 
        lstm_states = deepcopy(model._last_lstm_states)
        while env_steps < cfg.learn.total_timesteps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(last_obs, model.device)
                episode_starts = th.tensor(_last_episode_starts).float().to(model.device)
                # actions, values, log_probs, lstm_states = model.policy(obs_tensor, lstm_states, episode_starts)
                actions, values, log_probs, lstm_states = model.policy.forward(obs_tensor, lstm_states, episode_starts)
            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            env_steps += n_envs
            for info in infos:
                if 'episode' in info.keys():
                    ep_info = info['episode']
                    if len(ep_info.get('past_rewards', [])) > 0:
                        past_rew.append(ep_info['past_rewards'])
                    final_rs.append(ep_info['r'])
                    final_ts.append(ep_info['l'])
            # for idx, done_ in enumerate(dones):
            #     episode_starts[idx] = done_
            #     if done_: 
            #         #print('env index {} done, step {}'.format(idx, n_steps), rewards[idx]) #, infos)
            #         eps_rews.append(rewards[idx])
            last_obs = new_obs
            
            if env_steps % (256 * 256) == 0:
                tolog = {'Env Steps': env_steps}
                if len(past_rew) > 0:
                    past_rew = np.array(past_rew)    
                    for trial, (mean, std) in enumerate(zip(past_rew.mean(axis=0), past_rew.std(axis=0))):
                        tolog[f"rollout/trial{trial}/rew_mean"] =  mean 
                        tolog[f"rollout/trial{trial}/rew_std"] = std 
                    past_rew = []
                tolog['rollout/reward_mean'] = np.mean(final_rs)
                tolog['rollout/reward_std'] = np.std(final_rs)
                tolog['rollout/ep_len_mean'] = np.mean(final_ts)
                tolog['rollout/ep_len_std'] = np.std(final_ts)
                final_rs, final_ts = [], []
                
                if cfg.log_wb:
                    wandb.log(tolog)
                else:
                    print(tolog)

    
        

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None: 
    if cfg.eval_only:
        print('Running offline eval!')
        evaluate(cfg)
        return
    if cfg.eval_rl2:
        evaluate_rl2(cfg)
        return
    logging.info('Setting up environments %s', cfg.env_type)
    # set up env
    if cfg.env_type == 'procgen':
        if cfg.use_custom:
            cfg.env = cfg.procgen_custom 
            logging.info('Using custom procgen env! Max number of trials: %d' % cfg.procgen_custom.train.max_trials)
            env = SubprocVecEnv([
                lambda : make_custom_env(cfg.env.train) for i in range(cfg.env.num_train_env)])
            eval_env =  SubprocVecEnv([
                lambda : make_custom_env(cfg.env.eval) for i in range(cfg.env.num_eval_env)])
            n_envs = cfg.env.num_train_env
        else:
            cfg.env = cfg.procgen 
            env = ProcgenEnv(**(cfg.env.train))
            eval_env = ProcgenEnv(**(cfg.env.eval))
            n_envs = cfg.env.train.num_envs
        
        logging.info(
        f"Using Env: {cfg.env.name}, " + \
        f"Train on levels {cfg.env.train.start_level} - {cfg.env.train.start_level + cfg.env.train.num_levels}, " + \
        f"Eval on levels {cfg.env.eval.start_level} - {cfg.env.eval.start_level + cfg.env.eval.num_levels}"
        )
    elif cfg.env_type == 'atari':
        cfg.env = cfg.atari 
        env_ids = [name+'NoFrameskip-v4' for name in cfg.atari.env_ids]
        env = make_multitask_atari_env(
            env_ids=env_ids, n_envs=cfg.atari.n_envs, 
            reset_action_space=cfg.atari.reset_action_space,
            vec_env_cls=(SubprocVecEnv if cfg.atari.use_subproc_vec_env else DummyVecEnv),
            wrapper_kwargs=cfg.atari.wrapper_kwargs)
        env = VecFrameStack(env, n_stack=4)

        eval_env = make_multitask_atari_env(
            env_ids=env_ids, n_envs=int(len(env_ids) * 8), 
            reset_action_space=cfg.atari.reset_action_space,
            vec_env_cls=(SubprocVecEnv if cfg.atari.use_subproc_vec_env else DummyVecEnv),
            wrapper_kwargs=cfg.atari.eval_wrapper_kwargs
            ) #SubprocVecEnv)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        n_envs = cfg.atari.n_envs
    else:
        raise NotImplementedError('Unknown env type: %s' % cfg.env_type)
    
    if cfg.env_type == 'procgen' and cfg.use_custom:
        env = VecMonitor(env, info_keywords=["past_rewards"])
    else:
        env = VecMonitor(env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)

    cfg.run_name += f'-{n_envs}Envs-{cfg.ppo.batch_size}Batch'
    # set up log path
    log_path = join(os.getcwd(), 'log', cfg.run_name)
    os.makedirs(log_path, exist_ok=True)
    seed = len(list(filter(lambda x: 'seed' in x, os.listdir(log_path)))) + 1 # next_seed
    
    log_path = join(log_path, 'seed%d' % seed)
    os.makedirs(log_path, exist_ok=True) 
    os.makedirs(join(log_path, 'eval'), exist_ok=True) 
    cfg.log_path = log_path 
    
    cfg.learn.eval_log_path = join(log_path, 'eval')
    logging.info('Logging to:' + log_path)

    OmegaConf.save(cfg, join(log_path, 'config.yaml'))
    
    model = PPO(env=env, **cfg.ppo)
    if cfg.load_run != '':
        toload = join('/home/mandi/stable-baselines3/meta-rl/log', cfg.load_run)
        toload = join(toload, f'eval/models/{cfg.load_step}')
        model = model.load(env=env, path=toload) #'/home/mandi/stable-baselines3/log/log/burn/seed6/eval/best_model.zip')
        print('loaded model:', toload)
        print('Setting Reptile-k step to 0 for loaded model')
        model.reptile_k = 0

    # create logger object
    strings = ['stdout']
    if cfg.log_wb:  
        wandb_cfg = make_wandb_config(cfg)
        run = wandb.init(
            name=cfg.run_name, 
            config=wandb_cfg,
            **cfg.wandb
            )
        strings.append('wandb')
    logger = configure_logger(
        folder=log_path, 
        format_strings=strings,
        )
    model.set_logger(logger)
    logging.info('Created PPO policy')

    # create custom callback: evaluate env w.r.t. train update steps
    callback = [
        LogEvalCallback(
            eval_env=eval_env,
            n_eval_episodes=cfg.learn.n_eval_episodes,
            eval_freq=cfg.learn.eval_freq,
            log_path=log_path,
            model_save_path=cfg.learn.eval_log_path,
            verbose=cfg.vb
            )
        ] 

    if cfg.buffer_normalize:
        assert cfg.env_type == 'atari', 'Buffer normalization only works with atari'
        task_to_envs = {task_i: [] for task_i in range(len(env_ids))}
        for i in range(n_envs):
            task_to_envs[int(i % len(env_ids))].append(i)

        callback.append(
            NormalizeBufferRewardCallback(
                rollout_buffer=model.rollout_buffer,
                task_to_envs=task_to_envs,
                subtract_mean=cfg.subtract_mean,
                verbose=cfg.vb
                )
            )

    # train model
    model.learn(
        callback=callback,
        eval_env=eval_env,
        **cfg.learn)
 



if __name__ == '__main__':
    main()
