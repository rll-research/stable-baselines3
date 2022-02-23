CPUS=100-150
ARCH=NatureCNN

RUN=LSTM-${ARCH}
taskset -c $CPUS python train.py run_name=$RUN \
    ppo.policy_kwargs.features_extractor_kwargs.cnn_arch=${ARCH}  

