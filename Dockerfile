FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  wget \
  git \
  bzip2 \
  libx11-6 \
  libgomp1 \
  git

RUN apt-get install -y \
  python3.6 \
  python3.6-dev \
  python3.6-venv

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN rm get-pip.py

RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
RUN python3 --version
RUN $(head -1 `which pip3` | tail -c +3) --version

RUN rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

USER root
RUN adduser --disabled-password --gecos '' --shel /bin/bash user \
  && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90.-user

ENV HOME=/home/user
RUN chmod 777 /home/user

RUN ln -s /usr/local/cuda /usr/local/cuda-9.0
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64

RUN pip3 install numpy scipy matplotlib bs4 mido music21 tqdm argparse pillow requests
RUN pip3 install tensorflow-gpu==1.12.0 tensorboard==1.12.0
RUN pip3 install torch torchvision tensorboardX

USER user

RUN git clone https://github.com/mitmedialab/emb96.git
RUN cd emb96

EXPOSE 6006
