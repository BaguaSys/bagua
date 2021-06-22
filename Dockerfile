FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y curl software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN add-apt-repository ppa:git-core/ppa && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y git cmake mpich
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH=/root/.cargo/bin:${PATH}
RUN cargo install mdbook mdbook-linkcheck mdbook-katex mdbook-open-on-gh

RUN yes | python3 -m pip install -U setuptools wheel build pip

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/local/lib64:/usr/local/lib"
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/local/lib64:/usr/local/lib"
ENV PKG_CONFIG_PATH="/usr/local/cuda/pkgconfig/"
ENV CUDA_LIBRARY_PATH="/usr/local/cuda/lib64/"
