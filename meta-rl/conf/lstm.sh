CPUS=100-150
ARCH=NatureCNN

RUN=LSTM-${ARCH}
taskset -c $CPUS python train.py run_name=$RUN recurrent=True \
    cnn=${ARCH} 


CPUS=0-50
ARCH=NatureCNN

RUN=LSTM-2x128-${ARCH}-Hard100
taskset -c $CPUS python train.py run_name=$RUN recurrent=True \
    cnn=${ARCH}  \

     

