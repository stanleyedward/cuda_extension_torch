```sh
conda create -n cud python=3.8
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install cuda -c nvidia/label/cuda-11.3.0
```