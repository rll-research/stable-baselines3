# train: 100, 10_00, 10_000 levels, train for 100M steps

# Reptile
CPUS=0-64
taskset -c $CPUS python train.py run_name=Reptile3-ImpalaCNN-1e2levels \
    procgen.train.num_levels=100 \
    learn.total_timesteps=100e6   ppo.reptile_k=3

CPUS=32-64
taskset -c $CPUS python train.py run_name=Reptile3-ImpalaCNN-1e3levels \
    procgen.train.num_levels=1000 \
    learn.total_timesteps=100e6   ppo.reptile_k=3

CPUS=64-96
taskset -c $CPUS python train.py run_name=Reptile3-ImpalaCNN-1e4levels \
    procgen.train.num_levels=10000 \
    learn.total_timesteps=100e6   ppo.reptile_k=3

# RL2
CPUS=0-64
taskset -c $CPUS python train.py run_name=RL2-ImpalaCNN-1e2levels \
    procgen.train.num_levels=100 \
    learn.total_timesteps=100e6 recurrent=True use_custom=True \
    ppo.n_steps=512 policy_cfg=lstm ppo.buffer_sample_strategy=per_env \
    procgen_custom.num_train_env=1 ppo.batch_size=512 log_wb=False procgen_custom.num_eval_env=1


# reptile FT

LOAD_STEP=1500
for LOAD_LEVEL in 1e2 1e3 1e4
do
LOAD_RUN=Reptile3-ImpalaCNN-${LOAD_LEVEL}levels-256Envs-2048Batch/seed1
for LEVEL in {10000..10020}
do 
for SEED in 0
do 
RUN=FineTune-Reptile-${LOAD_LEVEL}-Unseen${LEVEL} 
taskset -c $CPUS python train.py run_name=$RUN \
        load_run=$LOAD_RUN load_step=$LOAD_STEP \
        procgen.train.start_level=${LEVEL} procgen.train.num_levels=1 \
        learn.total_timesteps=2e6 procgen.eval.start_level=${LEVEL} procgen.eval.num_levels=1 \
        learn.eval_freq=1 learn.log_interval=1 \
        procgen.eval.num_envs=50 learn.n_eval_episodes=100 ppo.seed=$SEED
done
done 
done 

# pretrain
CPUS=96-128
taskset -c $CPUS python train.py run_name=ImpalaCNN-1e2levels \
    procgen.train.num_levels=100 ;

CPUS=128-160
taskset -c $CPUS python train.py run_name=ImpalaCNN-1e3levels \
    procgen.train.num_levels=1000 
    
CPUS=160-192
taskset -c $CPUS python train.py run_name=ImpalaCNN-1e4levels \
    procgen.train.num_levels=10000 


# finetune: unseen levels in [0, 2, 7, 9, 15] + 10000

LOAD_STEP=1500
for LOAD_LEVEL in 1e2 1e3 1e4
do
LOAD_RUN=ImpalaCNN-${LOAD_LEVEL}levels-256Envs-2048Batch/seed1 
for LEVEL in {10000..10020}
do 
for SEED in 0
do 
RUN=FineTune-1e2-Unseen${LEVEL} 
taskset -c $CPUS python train.py run_name=$RUN \
        load_run=$LOAD_RUN load_step=$LOAD_STEP \
        procgen.train.start_level=${LEVEL} procgen.train.num_levels=1 \
        learn.total_timesteps=2e6 procgen.eval.start_level=${LEVEL} procgen.eval.num_levels=1 \
        learn.eval_freq=1 learn.log_interval=1 \
        procgen.eval.num_envs=50 learn.n_eval_episodes=100 ppo.seed=$SEED
done
done 
done 

 
# scratch  
for LEVEL in  {10010..10020}
do 
for SEED in 0 # 123 312 231 # 321 213
do
RUN=Scratch-Unseen${LEVEL}-Seed${SEED}
taskset -c $CPUS python train.py run_name=$RUN ppo.seed=${SEED} \
    procgen.train.start_level=${LEVEL} procgen.train.num_levels=1 \
    learn.total_timesteps=2e6 procgen.eval.start_level=${LEVEL} procgen.eval.num_envs=50 \
    procgen.eval.num_levels=1 \
    learn.eval_freq=1 learn.log_interval=1  learn.n_eval_episodes=100
done 
done 
# scratch but with NatureCNN
for LEVEL in  {10010..10020}
do 
for SEED in 0 # 123 312 231 # 321 213
do
RUN=Scratch-NatureCNN-Unseen${LEVEL}-Seed${SEED}
taskset -c $CPUS python train.py run_name=$RUN ppo.seed=${SEED} \
    procgen.train.start_level=${LEVEL} procgen.train.num_levels=1 \
    learn.total_timesteps=2e6 procgen.eval.start_level=${LEVEL} procgen.eval.num_envs=50 \
    procgen.eval.num_levels=1 cnn=NatureCNN \
    learn.eval_freq=1 learn.log_interval=1  learn.n_eval_episodes=100
done 
done 
 
 
TrainLev=1e4
LOAD_RUN=ImpalaCNN-${TrainLev}levels-256Envs-2048Batch/seed1 
LOAD_STEP=1500
for LEVEL in {10020..10000}
do
for SEED in 123 312 231 321 213
do
RUN=FineTune-${TrainLev}-Unseen${LEVEL}-Seed${SEED}
taskset -c $CPUS python train.py run_name=$RUN ppo.seed=${SEED} \
    load_run=$LOAD_RUN load_step=$LOAD_STEP \
    procgen.train.start_level=${LEVEL} procgen.train.num_levels=1 \
    learn.total_timesteps=2e6 procgen.eval.start_level=${LEVEL} procgen.eval.num_envs=50 procgen.eval.num_levels=1 \
    learn.eval_freq=1 learn.log_interval=1  learn.n_eval_episodes=100
done 
done 


# eval only
LOAD=Scratch-Unseen10016-Seed123-256Envs-2048Batch/seed1
RUN=EvalOnly
python train.py run_name=$RUN eval_only=True \
    load_run=$LOAD procgen.eval.start_level=10016 procgen.eval.num_levels=1 \
    procgen.eval.num_envs=100 learn.n_eval_episodes=100

# eval scratch
for LEVEL in {10000..10020}
do 
for SEED in  123 312 231 # 321 213
do
LOAD=Scratch-Unseen${LEVEL}-Seed${SEED}-256Envs-2048Batch/seed1
RUN=Eval-Scratch-${LEVEL}-Seed${SEED}
python train.py run_name=$RUN eval_only=True \
    load_run=$LOAD procgen.eval.start_level=${LEVEL} procgen.eval.num_levels=1 \
    procgen.eval.num_envs=100 learn.n_eval_episodes=100
 
done
done

# eval ft
for FT in FineTune-1e2 FineTune-1e3 FineTune-1e4
do
for LEVEL in {10000..10020}
do 
for SEED in  123 312 231 321 213
do 
LOAD=${FT}-Unseen${LEVEL}-Seed${SEED}-256Envs-2048Batch/seed1
RUN=Eval-${FT}-${LEVEL}-Seed${SEED}
python train.py run_name=$RUN eval_only=True \
    load_run=$LOAD procgen.eval.start_level=${LEVEL} procgen.eval.num_levels=1 \
    procgen.eval.num_envs=100 learn.n_eval_episodes=100
 
done
done
done 
