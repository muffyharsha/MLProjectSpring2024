#!/bin/sh
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
tar -xf Python-3.8.10.tgz
cd Python-3.8.10
./configure --enable-optimizations
make -j 8
sudo make altinstall
cd ..
rm Python-3.8.10.tgz

sudo python3.8 -m pip install pandas
sudo python3.8 -m pip install numpy
sudo python3.8 -m pip install pytz
sudo python3.8 -m pip install pyotp
sudo python3.8 -m pip install requests
sudo python3.8 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
sudo python3.8 -m pip install scikit-learn
sudo python3.8 -m pip install fastai
sudo python3.8 -m pip install fastcore
sudo python3.8 -m pip install matplotlib
sudo python3.8 -m pip install logging
