## Install on RLL server

```
conda create -n sb3 python=3.8
conda activate sb3
pip install torch==1.10.2
pip install -e . 
pip install wandb einops hydra-core procgen 
    
```