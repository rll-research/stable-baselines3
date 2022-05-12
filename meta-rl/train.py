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

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None: 
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
    
    env = VecMonitor(env)
    eval_env = VecMonitor(eval_env)

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

    # create logger object
    strings = ['stdout']
    if cfg.log_wb:  
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
