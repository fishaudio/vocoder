FROM nvcr.io/nvidia/pytorch:23.06-py3

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install code server and zsh
RUN wget -c https://github.com/coder/code-server/releases/download/v4.5.1/code-server_4.5.1_amd64.deb && \
    dpkg -i ./code-server_4.5.1_amd64.deb && \
    code-server --install-extension ms-python.python && \
    rm ./code-server_4.5.1_amd64.deb && \
    sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended

# Install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.1.0-beta.1/s5cmd_2.1.0-beta.1_Linux-64bit.tar.gz | tar xvz -C /tmp && \
    mv /tmp/s5cmd /usr/local/bin/s5cmd && s5cmd --help

# Install dependencies
WORKDIR /root/exp
COPY pyproject.toml .
RUN pip3 install .

CMD ["code-server", "--auth", "password", "--bind-addr", "0.0.0.0:8888", "/root/exp"]
