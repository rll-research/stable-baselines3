# train: 100, 10_00, 10_000 levels, train for 100M steps

# Reptile
CPUS=0-64
taskset -c $CPUS python train.py run_name=Reptile-ImpalaCNN-1e2levels \
    procgen.train.num_levels=100 \
    learn.total_timesteps=100e6   
