from procgen.env import ProcgenGym3Env, ToBaselinesVecEnv, ProcgenEnv
import random 
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from gym3 import vectorize_gym, ToGymEnv, ViewerWrapper, ExtractDictObWrapper
import gym 
from gym import spaces 
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from collections import deque 

class MultiProcGenEnv(gym.Env):
    def __init__(
        self, 
        env_name='coinrun',
        max_trials=3,
        reward_scales=None,
        num_levels=1000,
        start_level=0,
        distribution_mode='hard',
        is_train=True,
        restrict_themes=False,
    ): 
        self.max_trials = max_trials
        #if not self.train_mode:
        #    self.episode_len = 10
        self.reward_scales = reward_scales
        self.env_name = f"procgen:procgen-{env_name}-v0"
        self.num_levels = num_levels
        self.start_level = start_level
        self.distribution_mode = distribution_mode

        self.trial_num = 0
        self.restrict_themes = restrict_themes
 

        self.env = gym.make(
            self.env_name,
            num_levels=1,
            start_level=self.start_level,
            distribution_mode=self.distribution_mode,
            restrict_themes=self.restrict_themes,  
            )

        # All envs have the same action and obs space
        # action shape: (15,) - Discrete
        # obs shape: (64, 64, 3) - RGB
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict(
            {
                # 'rgb': spaces.Box(low=-25.0, high=255.0, shape=(65, 64, 3)),
                'rgb': spaces.Box(low=0, high=255, shape=(64, 64, 3)),
                'rew': spaces.Box(low=0.0, high=10.0 * (max_trials+1), shape=(1,)),
                'done': spaces.Box(low=0.0, high=1.0, shape=(1,)),
                'action': spaces.Box(low=0.0, high=1.0, shape=(15,)), # one-hot
                }
        )
        # self.observation_space = spaces.Dict(
        #     {'rgb': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)}
        # )
        # self.observation_space = spaces.Dict({
        #     'rgb': spaces.Box(low=0, high=255, shape=(64, 64, 3)),
        #     'prev_action': spaces.Box(low=0, high=15, shape=(1,)),
        #     'reward': spaces.Box(low=-1, high=1, shape=(1,)),
        #     'done': 
        # })
        self.trial_done = False
        self.is_train = is_train
        self.past_rewards = deque(maxlen=max_trials) # note max_trails is 0-indexed
 

    def reset(self):
        self.env.close()
        self.curr_level = random.randint(self.start_level, self.start_level + self.num_levels )
        self.env = gym.make(
            self.env_name, num_levels=1, start_level=self.curr_level, distribution_mode=self.distribution_mode)

        self.trial_done = False

        curr_obs = self.env.reset()
        
        # 4 is no action in procgen
        extras = np.zeros((1, 64, 3), dtype=np.float32)
        extras[0, :] = [4, 0.0, 0.0]
        cust_obs = curr_obs # np.concatenate([curr_obs/255.0, extras], axis=0)
        return {
                'rgb': cust_obs,
                'rew': 0.0,
                'done': 0.0,
                'action': np.eye(15)[4]
                }

    def seed(self, seed):
        return 

    def step(self, action):
        if self.trial_done:
            self.trial_done = False   
            next_obs = self.env.reset() 
            #extras = np.zeros((1, 64, 3), dtype=np.float32)
            #extras[0, :] = [4, 0.0, 0.0]
            cust_obs = next_obs # np.concatenate([next_obs/255.0, extras], axis=0)
            info = {'trial_num': self.trial_num}
            obs_dict = {
                'rgb': cust_obs,
                'rew': 0.0,
                'done': 0.0,
                'action': np.eye(15)[4],
                }
            return obs_dict, 0.0, False, {}
            # return {'rgb': next_obs}, 0.0, False, info

        next_obs, reward, done, info = self.env.step(action)

        # extras = np.zeros((1, 64, 3), dtype=np.float32)
        # extras[0, :] = [action, reward, float(done)]
        cust_obs = next_obs # np.concatenate([next_obs/255.0, extras], axis=0)
        
        if self.reward_scales != None:
            reward *= self.reward_scales 
        
        # if not self.is_train:
        #     if self.trial_num == self.max_trials:
        #         reward = reward
        #     else:
        #         reward = 0.0

        if done:
            if self.trial_num < self.max_trials:
                done = False
                self.trial_num += 1
                self.trial_done = True 
                self.past_rewards.append(reward)
                reward = 0 if self.is_train else reward
            else:
                self.trial_num = 0 
        
        info['trial_num'] = self.trial_num
        info['past_rewards'] = list(self.past_rewards)  
        return {'rgb': cust_obs, 'rew': reward, 'done': float(done), 'action': np.eye(15)[int(action)]}, reward, done, info
        # return {'rgb': next_obs}, reward, done, info

    def render(self, render_mode=None):
        return self.env.render(mode=render_mode)

class RepeatTrialProcgenGym3Env(ProcgenGym3Env):
    def __init__(
        self,
        max_trials=1, # for RL2!
        **kwargs,
    ):  
        kwargs['num_envs'] = 1 # 
        self.max_trials = 0 
        self.num_trials = 0
        self.start_level = kwargs.pop('start_level', 0)
        self.num_levels = kwargs.pop('num_levels', 1)
        self.current_level = random.randint(self.start_level, self.start_level + self.num_levels)
        self.env_kwargs = kwargs
        kwargs['start_level'] = self.current_level
        kwargs['num_levels'] = 1 
        self.env = ProcgenEnv(**kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        super().__init__(**kwargs)
    
    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        # print(dones)
        if True in dones:
            print(dones)
        if dones.all(): 
            self.num_trials += 1
        if self.num_trials < self.max_trials:
            dones = [False] * len(dones)
        else: 
            self.current_level = random.randint(self.start_level, self.start_level + self.num_levels)
            self.env_kwargs['start_level'] = self.current_level
            self.env = ProcgenEnv(**self.env_kwargs)
        return observations, rewards, dones, infos

    
# # env = ToGymEnv
# env = vectorize_gym(num=3, use_subproc=False, env_kwargs={'id': "procgen:procgen-coinrun-v0"}) #env_fn=make_custom_env)
# print(env)
# env = MultiProcGenEnv()
def make_custom_env(
    env_kwargs,
):
    return MultiProcGenEnv(**env_kwargs)  # RepeatTrialProcgenEnv(max_trials, **env_kwargs)
# env = vectorize_gym(num=3, use_subproc=False, env_fn=make_custom_env, env_kwargs={'max_trials': 3})
# print(env)
# env.reset()
# env.step([1,1,1])


# for _ in range(100000):
#     env.step(np.ones((1,)))
#     if env.num_trials > 0:
#         print(env.num_trials)
if __name__ == "__main__":
    cfg = OmegaConf.load('/home/mandi/stable-baselines3/meta-rl/conf/config.yaml').procgen_custom.train
    cfg.start_level = 10016
    cfg.num_levels = 1
    cfg.max_trials = 3
    # cfg.num_envs = 2 
    #env = SubprocVecEnv(env_fns=[lambda : make_custom_env(cfg) for i in range(2)])
    print('Making env')
    env = MultiProcGenEnv(**cfg)
    done = False
    obs = env.reset()
    print('start stepping')
    step = 0
    while not done:
        act = np.random.randint(0, 15) #, size=(1,))
        obs, rew, done, info = env.step(act)
        step += 1
        if env.trial_done: 
            print(f'Done trial {env.trial_num} at step:', step, rew, info)
    print(f'Done episode {env.trial_num} at step:', step, rew, info)
    env.close()
    #print(env.reset().shape)