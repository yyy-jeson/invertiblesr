


####

#### 运行编码工具

```
unzip tools.zip;mv tools/* .
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/cpfs-data/invsr/codes
```
##### av库安装
```
apt install -y python-dev python-virtualevn pkg-config
apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev
python3 -m pip install av==8.0.2
``` 
##### requiremtn
```
pip install scikit-learn
python3 -m pip install -U --no-cache torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple 
```