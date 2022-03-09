"""
install gym atari first! also cv2 is needed 
- pip install gym[atari] gym[accept-rom-license] opencv-python
"""
import gym
from PIL import Image 
from typing import Any, Callable, Dict, Optional, Type, Union, List
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

def make_multitask_atari_env(
    env_ids: Union[List[str], List[Type[gym.Env]]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = AtariWrapper,
    env_kwargs: Optional[Dict[str, Any]] = {},
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = {},
    monitor_kwargs: Optional[Dict[str, Any]] = {},
    wrapper_kwargs: Optional[Dict[str, Any]] = {}
) -> VecEnv:

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapper(env, **wrapper_kwargs)
        return env
    n_games = len(env_ids)
    assert n_envs >= n_games, 'must make sure each game has at least one env'
    

    def make_env(rank):
        env_id = env_ids[int(rank % n_games)]
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init
    
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    
    

def main():
    games = ['Breakout', 'BeamRider', 'Seaquest', 'Asteroids'] # ['Pong', 'BeamRider', 'Breakout'] # , 'Enduro', 'Qbert', 'Seaquest', 'SpaceInvaders', 'Asteroids', 'RoadRunner']
    env_ids = [game+'NoFrameskip-v4' for game in games] #	Pong	Qbert	Seaquest	SpaceInvaders
    n_envs = len(games)
    print('making env')
    env = make_multitask_atari_env(env_ids, n_envs, wrapper_kwargs={'screen_size': 200})
    print('done')
    obs = env.reset()
    print(type(obs), obs.shape)
    print(env.action_space)
    actions = np.array([env.action_space.sample() for _ in range(n_envs)])
    obs = env.step(actions)
    print('after step', type(obs), obs[0].shape, )

    games = ['Pong']
    env_ids = [game+'NoFrameskip-v4' for game in games] 
    env = make_multitask_atari_env(env_ids, n_envs, wrapper_kwargs={'screen_size': 100})
    obs = env.reset()
    actions = np.array([np.zeros(8, dtype=np.int) for _ in range(n_envs)])
    obs = env.step(actions)
    print('after step', type(obs), obs[0].shape, )
    # fig, axs = plt.subplots(1, n_envs, figsize=(10*n_envs, 10))
    # for i, ax in enumerate(axs):
    #     ax.imshow(obs[i], cmap='gray')
    #     ax.set_title(games[i], fontsize=60)
    #     ax.axis('off')
    # plt.savefig('atari_env.png')
     

if __name__ == "__main__":
    main() 