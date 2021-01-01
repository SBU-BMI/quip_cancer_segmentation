FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
	pip install --upgrade pip && \
	conda update -n base -c defaults conda && \
	pip3 install setuptools==45 && \
	pip install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
	conda install --yes scikit-learn && \
	pip install "Pillow<7" pymongo pandas && \
	pip install torchvision==0.2.1 && \
	conda install --yes -c conda-forge opencv

RUN	pip install openslide-python

COPY	. /root/quip_cancer_segmentation/.

RUN	cd /root/quip_cancer_segmentation/models_cnn && \
	wget --no-check-certificate -v -O models.zip -L \
		https://stonybrookmedicine.box.com/shared/static/1hdfb06lgd08xfbpoly9tjp6c6i665nz.zip >/dev/null 2>&1 && \
        unzip -o models.zip && rm -f models.zip && mv brca_models_cnn/* . && rm -rf brca_models_cnn && \
		chmod 0755 /root/quip_cancer_segmentation/scripts/*

ENV	BASE_DIR="/root/quip_cancer_segmentation"
ENV	PATH="./":$PATH
WORKDIR	/root/quip_cancer_segmentation/scripts

CMD ["/bin/bash"]
