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

# 4 tasks:
ARCH=NatureCNN #d ImpalaCNN
RUN=NoPong-$ARCH-64Envs-B512-NoScaleRew-NoVFClip
taskset -c $CPUS python train.py run_name=$RUN env=${atari} env_type=atari learn.total_timesteps=1e8 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=512  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
 cnn=$ARCH policy_cfg=ImgObs atari.n_envs=64

# fine tune on Pong:
LOAD=NoPong-Scratch-NatureCNN-64Envs-NoScaleRew/seed3
LOAD_STEP=300
RUN=Pong-Finetune-$ARCH-64Envs-B256-NoVFClip
taskset -c $CPUS python train.py run_name=$RUN env_type=atari learn.total_timesteps=1e8 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
 cnn=$ARCH policy_cfg=ImgObs atari.n_envs=64 load_run=$LOAD load_step=$LOAD_STEP \
 atari.env_ids=[Pong] atari.reset_action_space=18



# 1task
ARCH=NatureCNN #d ImpalaCNN
RUN=1Task-Pong-Scratch-$ARCH
CPUS=100-140
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[Pong] \
env_type=atari learn.total_timesteps=1e7 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs

ARCH=NatureCNN #d ImpalaCNN
RUN=1Task-Pong-Scratch-$ARCH-32Envs
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[Pong] \
env_type=atari learn.total_timesteps=1e7 \
ppo.n_steps=128 ppo.n_epochs=4 ppo.batch_size=256  \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
 cnn=$ARCH policy_cfg=ImgObs atari.n_envs=128 
 





ARCH=NatureCNN #d ImpalaCNN
NAME=Asteroids
RUN=1Task-${NAME}-Scratch-$ARCH-64Envs-Bsize512
taskset -c $CPUS python train.py run_name=$RUN env=${atari} atari.env_ids=[${NAME}] \
env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 \
ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1  cnn=$ARCH policy_cfg=ImgObs \
atari.n_envs=64 ppo.batch_size=512  learn.total_timesteps=1e9


# try: normalize reward by max
NAME=Asteroids
RUN=ScaleRew-${NAME}-Scratch-NatureCNN-32Envs-Bsize512-Sub
taskset -c $CPUS python train.py run_name=$RUN atari.env_ids=[${NAME}] env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 cnn=NatureCNN policy_cfg=ImgObs \
ppo.batch_size=512 learn.total_timesteps=1e9 atari.n_envs=32 


atari.scale_reward=True 