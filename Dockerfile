# FROM tensorflow/tensorflow:1.13.1-gpu-py3
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04
LABEL maintainer "Luc Michalski <lmichalski@evolutive-business.com>"

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      python3 \
      python3-pip \
      python3-dev \
      python3-setuptools \
      python3-cairo \
      python3-cairo-dev \
      python3-distutils-extra \
      libgirepository1.0-dev \
      libcairo-gobject2 \
      libcairo2-dev \
      build-essential \
      cmake \
      git \
      curl \
      libcurl4-openssl-dev \
      pkg-config \
      ca-certificates \
      libjpeg-dev \
      libpng-dev \
      libssl-dev \ 
      openmpi-bin \
      screen \
      unrar \
      unzip \
      wget && \
    rm -rf /var/lib/apt/lists/* \
    apt-get upgrade

WORKDIR /src

RUN python3 -m pip install --upgrade pip

ADD src/requirements.txt /src/requirements.txt

RUN pip3 install --no-cache -r requirements.txt

COPY ./src .
COPY ./docker-entrypoint.sh .

EXPOSE 5000
VOLUME ["/src/uploads", "/src/weights", "/src/results"]

# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/

RUN echo $LD_LIBRARY_PATH && \
    ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10.2.2.89 /usr/lib/x86_64-linux-gnu/libcublas.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so.10.3.0.89 /usr/lib/x86_64-linux-gnu/libcusolver.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2.89 /usr/lib/x86_64-linux-gnu/libcudart.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcurand.so.10.1.2.89  /usr/lib/x86_64-linux-gnu/libcurand.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusparse.so.10.3.1.89  /usr/lib/x86_64-linux-gnu/libcusparse.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcufft.so.10.1.2.89  /usr/lib/x86_64-linux-gnu/libcufft.so.10.0

RUN apt-get update && apt-get install -y mlocate && updatedb

ENTRYPOINT ["./docker-entrypoint.sh"]
# ENTRYPOINT ["python3"]
# CMD ["app.py"]