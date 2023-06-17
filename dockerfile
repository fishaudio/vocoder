FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install jupyter-lab and zsh
RUN pip3 install jupyterlab && \
    sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended

# Install dependencies
WORKDIR /root/exp
COPY pyproject.toml .
RUN pip3 install .

CMD ["jupyter", "lab", "--ip=*", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
