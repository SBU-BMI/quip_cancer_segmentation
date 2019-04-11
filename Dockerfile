FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
MAINTAINER HAN LE

RUN 	apt-get -y update
RUN   pip install --upgrade pip && \
      conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
      conda install --yes -c conda-forge opencv && \
      conda install --yes scikit-learn && \
      pip install --yes Pillow && \
      pip install --yes torchvision && \
      pip install --yes openslide-python

CMD ["/bin/bash"]
