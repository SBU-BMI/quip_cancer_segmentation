FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
MAINTAINER Tahsin Kurc

RUN 	apt-get -y update
RUN   pip install --upgrade pip && \
      conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
      conda install --yes scikit-learn && \
      pip install Pillow && \
      pip install torchvision==0.2.1 && \
      apt-get install --yes python3-openslide && \
      pip install openslide-python && \
      conda install --yes -c conda-forge opencv

CMD ["/bin/bash"]
