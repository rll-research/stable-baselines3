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
from custom.mt_atari import make_multitask_atari_env



def main():
    env = make_multitask_atari_env(
        env_ids=['AsteroidsNoFrameskip-v4'], n_envs=8, wrapper_kwargs={'terminal_on_life_loss': False})
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)
    obs = env.reset()
    for i in range(10000):
        clipped_actions = [2 for _ in range(8)]
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        env.reset()
        if i % 1000 == 0:
            print(f"taking step {i}")
    print('done stepping')
    return

if __name__ == '__main__':
    main()
