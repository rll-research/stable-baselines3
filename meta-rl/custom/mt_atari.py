"""
install gym atari first! also cv2 is needed 
- pip install gym[atari] gym[accept-rom-license] opencv-python
"""
import gym
from PIL import Image 
from typing import Any, Callable, Dict, Optional, Type, Union, List
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.atari_wrappers import * # all the avaliable atari wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

MAX_REWARDS = {
    'Pong': 20,
    'BeamRider': 4295,
    'Breakout': 358,
    'Seaquest': 2000, 
    'Asteroids': 780,
    'Enduro': 830,
}

class MaskAction(gym.Wrapper):
    def __init__(self, env: gym.Env, handle_as: int = 0):
        gym.Wrapper.__init__(self, env)
        self.max_action = env.action_space.n
        self.handle_as = handle_as
        assert handle_as < self.max_action, 'Cannot replace invaid action with {}'.format(handle_as)

    def step(self, action: int) -> GymStepReturn:
        if action >= self.max_action:
            action = self.handle_as
        return self.env.step(action)

class ClipScaleRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env, env_id: str):
        gym.RewardWrapper.__init__(self, env)
        env_name = env_id
        if 'NoFrame' in env_name:
            env_name = env_id.split('NoFrameskip-v4')[0]
        assert env_name in MAX_REWARDS, f'{env_name} is not in {MAX_REWARDS}'
        self.max_reward = MAX_REWARDS[env_name]


    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward) / self.max_reward



class CustomAtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True, 
        scale_reward: bool = False, 
        env_id: str = "PongNoFrameskip-v4",
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        env = MaskAction(env)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings(): 
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        
        if clip_reward and scale_reward:
            env = ClipScaleRewardEnv(env, env_id)
        elif clip_reward:
            env = ClipRewardEnv(env)

        super(CustomAtariWrapper, self).__init__(env)


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
    wrapper_kwargs: Optional[Dict[str, Any]] = {},
    reset_action_space: int = -1, 
) -> VecEnv:

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
            # if monitor_path is not None:
            #    os.makedirs(monitor_dir, exist_ok=True)
            # env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            # print(env_id, 'FIRE?', "FIRE" in env.unwrapped.get_action_meanings())
            env = CustomAtariWrapper(env, **wrapper_kwargs)

            return env

        return _init
    
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    tmp_envs = [gym.make(env_id, **env_kwargs) for env_id in env_ids]
    max_action = max([env.action_space.n for env in tmp_envs])
    if reset_action_space > 0:
        max_action = reset_action_space

    env_fns = [make_env(i + start_index) for i in range(n_envs)] 
    vec_env = vec_env_cls(env_fns, **vec_env_kwargs)
    print('Reset action space range for envs: {}'.format(max_action), [env.action_space.n for env in tmp_envs])
    del tmp_envs

    vec_env.action_space = gym.spaces.Discrete(max_action)
    return vec_env
    
    

def main():
    games = ['Breakout', 'BeamRider', 'Seaquest', 'Asteroids', 'Pong'] # 'BeamRider', 'Breakout'] # , 'Enduro', 'Qbert', 'Seaquest', 'SpaceInvaders', 'Asteroids', 'RoadRunner']
    # env_ids = [game+'NoFrameskip-v4' for game in games] #	Pong	Qbert	Seaquest	SpaceInvaders
    n_envs = len(games)
    # print('making env')
    # env = make_multitask_atari_env(env_ids, n_envs, wrapper_kwargs={'screen_size': 200})
    # print('done')
    # obs = env.reset()
    # print(type(obs), obs.shape)
    # print(env.action_space)
    # actions = np.array([env.action_space.sample() for _ in range(n_envs)])
    # obs = env.step(actions)
    # print('after step', type(obs), obs[0].shape, )

    # games = ['Pong']
    env_ids = [game+'NoFrameskip-v4' for game in games] 
    env = make_multitask_atari_env(env_ids, n_envs, wrapper_kwargs={'screen_size': 100})
    obs = env.reset()
    actions = np.array([8 for _ in range(n_envs)])
    obs = env.step(actions)
    print('after step', type(obs), obs[0].shape, )
    # fig, axs = plt.subplots(1, n_envs, figsize=(10*n_envs, 10))
    # for i, ax in enumerate(axs):
    #     ax.imshow(obs[i], cmap='gray')
    #     ax.set_title(games[i], fontsize=60)
    #     ax.axis('off')
    # plt.savefig('atari_4env.png')
     

if __name__ == "__main__":
    main() 