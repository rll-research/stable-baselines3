CPUS=0-100
ARCH=ImpalaCNN
LEVEL=100
taskset -c $CPUS python train.py run_name=Hard-${LEVEL}-${ARCH}-AllTheme \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.num_levels=${LEVEL} 

# 1 theme
CPUS=50-100
ARCH=ImpalaCNN
LEVEL=100
taskset -c $CPUS python train.py run_name=Hard-${LEVEL}-${ARCH}-1Theme \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH} env.train.num_levels=${LEVEL} env.train.restrict_themes=True 