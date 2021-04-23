FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev vim && \
	pip install --upgrade pip && \
	conda update -n base -c defaults conda && \
	pip3 install setuptools==45 && \
	pip install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
	conda install --yes scikit-learn && \
	pip install Pillow pymongo && \
	pip install torchvision==0.2.1 && \
	conda install --yes -c conda-forge opencv

RUN 	pip3 install setuptools==45 && pip install openslide-python

COPY	. /root/ajp_til_analysis/.

RUN		chmod 0755 /root/ajp_til_analysis/scripts/*

ENV	BASE_DIR="/root/ajp_til_analysis"
ENV	PATH="./":$PATH
WORKDIR	/root/ajp_til_analysis/scripts

CMD ["/bin/bash"]
