PPO config from sb3-zoo:

atari:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  frame_stack: 4
  policy: 'CnnPolicy'
  n_envs: 8
  n_steps: 128
  n_epochs: 4
  batch_size: 256
  n_timesteps: !!float 1e7
  learning_rate: lin_2.5e-4
  clip_range: lin_0.1
  vf_coef: 0.5
  ent_coef: 0.01

ARCH=NatureCNN #d ImpalaCNN
RUN=Pong-Scratch-$ARCH
CPUS=100-140
taskset -c $CPUS python train.py run_name=$RUN env=${atari} env_type=atari learn.total_timesteps=1e7 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs

# 1task
ARCH=NatureCNN #d ImpalaCNN
RUN=1Task-Pong-Scratch-$ARCH
CPUS=100-140
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[Pong] \
env_type=atari learn.total_timesteps=1e7 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs

ARCH=NatureCNN #d ImpalaCNN
RUN=1Task-Pong-Scratch-$ARCH-128Envs
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[Pong] \
env_type=atari learn.total_timesteps=1e7 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs atari.n_envs=128

ARCH=NatureCNN #d ImpalaCNN
NAME=Asteroids
RUN=1Task-${NAME}-Scratch-$ARCH-128Envs-Bsize512
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[${NAME}] \
env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs \
atari.n_envs=128 ppo.batch_size=512  learn.total_timesteps=1e9


# try: normalize reward by max
NAME=Asteroids
RUN=ScaleRew-${NAME}-Scratch-NatureCNN-32Envs-Bsize512-Sub
taskset -c $CPUS python train.py run_name=$RUN atari.env_ids=[${NAME}] env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 cnn=NatureCNN policy_cfg=ImgObs \
ppo.batch_size=512 learn.total_timesteps=1e9 atari.n_envs=32 


atari.scale_reward=True 