CPUS=100-150
ARCH=NatureCNN

RUN=LSTM-${ARCH}
taskset -c $CPUS python train.py run_name=$RUN recurrent=True \
    cnn=${ARCH} 


CPUS=0-50
ARCH=NatureCNN
RUN=LSTM-2x128-lr1e-4-${ARCH}-Hard100
taskset -c $CPUS python train.py run_name=$RUN recurrent=True \
    cnn=${ARCH} ppo_lstm.policy_kwargs.lstm_hidden_size=128 


CPUS=50-100
ARCH=NatureCNN
RUN=LSTM-1x256-${ARCH}-Hard100
taskset -c $CPUS python train.py run_name=$RUN recurrent=True \
    cnn=${ARCH} ppo_lstm.policy_kwargs.n_lstm_layers=1

     
# 1theme!
CPUS=0-40 
RUN=LSTM-1Theme-1x256-ImpalaCNN-PerEnvSample-Custom1e5
taskset -c $CPUS python train.py run_name=$RUN recurrent=True ppo_lstm.policy_kwargs.lstm_hidden_size=256 \
ppo_lstm.policy_kwargs.n_lstm_layers=1 custom_env.train.max_trials=1 \
use_custom=True ppo_lstm.buffer_sample_strategy=per_env custom_env.train.restrict_themes=True custom_env.train.num_levels=1e5

# changed conat method! 
CPUS=60-100
RUN=NewConcat-LSTM-1Theme-1x256-ImpalaCNN-Custom1e2
taskset -c $CPUS python train.py run_name=$RUN recurrent=True ppo_lstm.policy_kwargs.lstm_hidden_size=256 \
ppo_lstm.policy_kwargs.n_lstm_layers=1 custom_env.train.max_trials=1 \
use_custom=True ppo_lstm.buffer_sample_strategy=per_env custom_env.train.restrict_themes=True custom_env.train.num_levels=1e2
