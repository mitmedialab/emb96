FROM nvidia/cuda:9.0-base-ubuntu16.04

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

RUN adduser --disabled-password --gecos '' --shel /bin/bash user \
  && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90.-user
USER user

ENV HOME=/hom/user
RUN chmod 777 /home/user

RUN sudo pip3 install numpy scipy matplotlib bs4 mido music21 tqdm argparse pillow
RUN sudo pip3 install tensorflow-gpu tensorboard
RUN sudo pip3 install torch torchvision tensorboardX

RUN git clone https://github.com/mitmedialab/emb96.git
RUN cd emb96

RUN mkdir simple
RUN tensorboard --logdir simple

RUN cd src
RUN python3 main.py \
  --dataset_dir ../dataset/ \
  --t_dataset_dir ../t_dataset/ \
  --generated_dir ../generated/ \
  --epochs 250 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --beta 4. \
  --num_workers 4 \
  --experience_name simple \
  --saving_rate 1 \
  --donwload --transpose --generate --train \
