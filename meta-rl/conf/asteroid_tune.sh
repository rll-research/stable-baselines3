# folloing sb-zoo config:
NAME=Asteroids
CPUS=0-100
RUN=Tune-${NAME}-NatureCNN
taskset -c $CPUS python train.py run_name=$RUN atari.env_ids=[${NAME}] env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
cnn=NatureCNN policy_cfg=ImgObs \
ppo.batch_size=256 learn.total_timesteps=1e7 atari.n_envs=8 cfg.atari.

NAME=Asteroids
CPUS=100-160
RUN=Tune-${NAME}-NatureCNN-NoEpisodic
taskset -c $CPUS python train.py run_name=$RUN atari.env_ids=[${NAME}] env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
cnn=NatureCNN policy_cfg=ImgObs \
ppo.batch_size=256 learn.total_timesteps=1e7 atari.n_envs=8 \
atari.wrapper_kwargs.terminal_on_life_loss=False

Breakout
NAME=BeamRider 
CPUS=160-220
RUN=Tune-${NAME}-NatureCNN-NoEpisodic
taskset -c $CPUS python train.py run_name=$RUN atari.env_ids=[${NAME}] env_type=atari \
ppo.n_steps=128 ppo.n_epochs=4 ppo.learning_rate=linear_2.5e-4 ppo.clip_range=linear_0.1 \
cnn=NatureCNN policy_cfg=ImgObs \
ppo.batch_size=256 learn.total_timesteps=1e7 atari.n_envs=8 \
atari.wrapper_kwargs.terminal_on_life_loss=False
