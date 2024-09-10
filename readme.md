env
```sh
conda create -n cud python=3.8
conda install conda-forge::gxx=9.5.0
conda install cuda -c nvidia/label/cuda-11.3.1
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

symlink gxx
```sh
ln -s /home/stanley/miniconda3/envs/torch112cu113/bin/gcc gcc
ln -s /home/stanley/miniconda3/envs/torch112cu113/bin/g++ g++
```