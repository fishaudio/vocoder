FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
WORKDIR /root/exp
COPY pyproject.toml .
RUN pip3 install .

# Install code-server and zsh
RUN wget -c https://github.com/coder/code-server/releases/download/v4.10.0/code-server_4.10.0_amd64.deb && \
    dpkg -i ./code-server_4.10.0_amd64.deb && \
    rm ./code-server_4.10.0_amd64.deb && \
    sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended
