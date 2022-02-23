import numpy as np
from procgen import ProcgenEnv
import matplotlib.pyplot as plt

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
ls = []
for i in range(10):
    env = ProcgenEnv(
    env_name='coinrun', 
    num_levels=1, 
    start_level=10010+i, 
    num_envs=1, 
    restrict_themes=False, 
    distribution_mode='hard',
    render_mode="rgb_array",
    )
    ls.append(env.reset()['rgb'][0])
    env.close()

imgs = np.concatenate(ls, axis=1)

plt.imsave('harder10.png', imgs)

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