CPUS=0-100
ARCH=ImpalaCNN

 

# load the all-theme NatureCNN-100-AllTheme model
CPUS=100-150
ARCH=NatureCNN
LOAD_RUN=Hard-100-NatureCNN-AllTheme/seed2
LOAD_STEP=1050 
RUN=FineTune-Unseen10-Hard-${ARCH}-AllTheme
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} \
    load_run=$LOAD_RUN load_step=$LOAD_STEP \
    env.train.start_level=10000 env.train.num_levels=10 learn.total_timesteps=5e6 env.eval.start_level=10000 env.eval.num_levels=10 \
    learn.eval_freq=1 learn.log_interval=1 

# compare to train from scratch! 
CPUS=200-255
ARCH=NatureCNN
RUN=Scratch-10-Hard-${ARCH}-AllTheme
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} \
    env.train.start_level=10000 env.train.num_levels=10 learn.total_timesteps=5e6 env.eval.start_level=10000 env.eval.num_levels=10 \
    learn.eval_freq=1 learn.log_interval=1 ;

# pabti5: 1e4 train

LOAD_RUN=Hard-10000-NatureCNN-1Theme/seed1 
LOAD_STEP=1500 
ARCH=NatureCNN
RUN=FineTune-Unseen10-Hard-${ARCH}-1Theme
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} \
    load_run=$LOAD_RUN load_step=$LOAD_STEP \
    env.train.start_level=10000 env.train.num_levels=10 learn.total_timesteps=5e6 env.eval.start_level=10000 env.eval.num_levels=10 \
    learn.eval_freq=1 learn.log_interval=1 

LOAD_RUN=Hard-10000-NatureCNN-AllTheme/seed1 
LOAD_STEP=550 

LOAD_RUN=Hard-10000-ImpalaCNN-AllTheme/seed1 
LOAD_STEP=1500 
RUN=FineTune-Unseen1-Hard-${ARCH}-AllTheme

ARCH=ImpalaCNN
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.start_level=10000 env.train.num_levels=1 \
    load_run=$LOAD_RUN load_step=$LOAD_STEP learn.eval_freq=1 learn.log_interval=1 learn.total_timesteps=5e6 \
    env.eval.start_level=10000 env.eval.num_levels=1

RUN=Scratch-Unseen1-Hard-${ARCH}-AllTheme
ARCH=ImpalaCNN
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.start_level=10000 env.train.num_levels=1 \
    learn.eval_freq=1 learn.log_interval=1 learn.total_timesteps=5e6 \
    env.eval.start_level=10000 env.eval.num_levels=1

ARCH=NatureCNN
RUN=Scratch-1e4-2e4-Hard-${ARCH}-AllTheme
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} \
    env.eval.start_level=10000 env.train.start_level=10000 env.train.num_levels=10000 \
    learn.eval_freq=5 learn.log_interval=1 learn.total_timesteps=5e6 



# 1 theme
CPUS=50-100
ARCH=ImpalaCNN
LEVEL=100
taskset -c $CPUS python train.py run_name=Hard-${LEVEL}-${ARCH}-1Theme \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.num_levels=${LEVEL} env.train.restrict_themes=True 


# 1000 on ti5 
CPUS=0-20
ARCH=NatureCNN
LEVEL=10000
taskset -c $CPUS python train.py run_name=Hard-${LEVEL}-${ARCH}-1Theme \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.num_levels=${LEVEL} env.train.restrict_themes=True 

CPUS=20-48
ARCH=NatureCNN
LEVEL=10000
taskset -c $CPUS python train.py run_name=Hard-${LEVEL}-${ARCH}-AllTheme \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.num_levels=${LEVEL}


# debug fine-tune 
RUN=Debug-0LearnRate-FineTune-ImpalaCNN-AllTheme 
python train.py run_name=$RUN ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=ImpalaCNN \
    env.train.start_level=0 env.train.num_levels=10000 load_run=Hard-10000-ImpalaCNN-AllTheme/seed1 \
    load_step=1500 learn.total_timesteps=1e6 env.eval.start_level=0 env.eval.num_levels=10000 ppo.learning_rate=0 

