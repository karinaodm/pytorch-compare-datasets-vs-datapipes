FROM nvcr.io/nvidia/pytorch:22.05-py3

RUN apt-get update -y \
    && apt install sshpass protobuf-compiler git git-lfs fontconfig debconf -y \ 
    && echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && DEBIAN_FRONTEND=noninteractive apt-get install ttf-mscorefonts-installer -y || echo "OK TO FAIL" \
    && sed -i 's#http://downloads.sourceforge.net/corefonts#https://github.com/pushcx/corefonts/raw/master#g' /usr/share/package-data-downloads/ttf-mscorefonts-installer \
    && /usr/lib/update-notifier/package-data-downloader \
    && fc-cache -fv

WORKDIR /working_dir
ADD requirements.txt /working_dir/requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends mc zbar-tools ffmpeg libsm6 libxext6
RUN python3 -m pip install -r /working_dir/requirements.txt

CMD ["/bin/bash"]
