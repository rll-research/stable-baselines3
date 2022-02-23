import os
import logging
from os.path import join
from stable_baselines3 import PPO
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
from custom.callbacks import LogEvalCallback
from custom.custom_procgen_env import MultiProcGenEnv, make_custom_env

@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:


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
 

    # set up env
    # env = ProcgenEnv(**(cfg.env.train))
    # env = VecMonitor(env)

    # eval_env = ProcgenEnv(**(cfg.env.eval))
    # eval_env = VecMonitor(eval_env)
    env = SubprocVecEnv([
        lambda : make_custom_env(cfg.custom_env.train) for i in range(cfg.custom_env.num_train_env)])
    env = VecMonitor(env)
    eval_env =  SubprocVecEnv([
        lambda : make_custom_env(cfg.custom_env.eval) for i in range(cfg.custom_env.num_eval_env)])
    eval_env = VecMonitor(eval_env)

    logging.info(
        f"Using Env: {cfg.env.name}, " + \
        f"Train on levels {cfg.env.train.start_level} - {cfg.env.train.start_level + cfg.env.train.num_levels}, " + \
        f"Eval on levels {cfg.env.eval.start_level} - {cfg.env.eval.start_level + cfg.env.eval.num_levels}"
        )


    model = PPO(env=env, **cfg.ppo)
    if cfg.load_run != '':
        toload = join('/home/mandi/stable-baselines3/meta-rl/log', cfg.load_run)
        toload = join(toload, f'eval/models/{cfg.load_step}')

        #print(model.policy.state_dict()['action_net.bias'])
        model = model.load(env=env, path=toload) #'/home/mandi/stable-baselines3/log/log/burn/seed6/eval/best_model.zip')
        print('loaded model:', toload)
        #print(model.policy.state_dict()['action_net.bias'])
        #raise ValueError 


    # create logger object
    strings = ['stdout']
    if cfg.log_wb:
        run = wandb.init(
            name=(log_path.split('/')[-2]), 
            config=dict(cfg),
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
    model.learn(
        callback=callback,
        eval_env=eval_env,
        **cfg.learn)
 



if __name__ == '__main__':
    main()
