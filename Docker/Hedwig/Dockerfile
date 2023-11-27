FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DISPLAY=:1

RUN DEBIAN_FRONTEND=noninteractive 
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ENV TZ=America/Bogota
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN sed -i'' 's/archive\.ubuntu\.com/us\.archive\.ubuntu\.com/' /etc/apt/sources.list
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends tzdata keyboard-configuration 
RUN apt-get install ffmpeg xvfb libsm6 libxext6 libxcb-* wget -y
RUN apt-get install git -y
RUN apt-get install -y locales
RUN locale-gen es_ES.UTF-8
RUN locale-gen en_US.UTF-8

COPY ./nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb
RUN dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb
RUN apt install tensorrt libnvinfer8 -y

COPY ./requirements.txt requirements.txt
RUN pip3 install -U -r requirements.txt --default-timeout=100

RUN pip install git+https://github.com/openai/CLIP.git 
RUN pip3 install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@v0.4.0
RUN pip3 install git+https://github.com/NVIDIA-AI-IOT/nanoowl.git

RUN useradd -m -u 1000 user
USER user

WORKDIR /workdir
COPY --chown=user ./main.py main.py
COPY --chown=user models models

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none"]