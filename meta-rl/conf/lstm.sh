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

     

