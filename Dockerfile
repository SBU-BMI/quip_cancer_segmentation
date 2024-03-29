FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
	pip install --upgrade pip==21.0.1 && \
	pip3 install setuptools==45 && \
	pip3 install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
	pip3 install scikit-learn && \
	pip3 install "Pillow<7" pymongo pandas && \
	pip3 install torchvision==0.2.1 && \
	pip3 install opencv-python && \
	pip3 install openslide-python

ENV     BASE_DIR="/quip_app/quip_cancer_segmentation"
ENV     PATH="./":$PATH

COPY	. ${BASE_DIR}/.

RUN	cd ${BASE_DIR}/models_cnn && \
	wget --no-check-certificate -v -O models.zip -L \
		https://stonybrookmedicine.box.com/shared/static/1hdfb06lgd08xfbpoly9tjp6c6i665nz.zip >/dev/null 2>&1 && \
        unzip -o models.zip && rm -f models.zip && mv brca_models_cnn/* . && rm -rf brca_models_cnn && \
	chmod 0755 ${BASE_DIR}/scripts/*

WORKDIR	${BASE_DIR}/scripts

CMD ["/bin/bash"]
