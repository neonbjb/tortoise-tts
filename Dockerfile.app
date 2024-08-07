FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS base

# Copy application code to /app
COPY . /app

# Install necessary packages and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HOME="/root"
ENV CONDA_DIR="${HOME}/miniconda"
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PIP_DOWNLOAD_CACHE="$HOME/.pip/cache"
ENV TORTOISE_MODELS_DIR="$HOME/tortoise-tts/build/lib/tortoise/models"

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniconda3.sh && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> "${HOME}/.bashrc"

FROM base AS conda_base

# --login option used to source bashrc (thus activating conda env) at every RUN statement
SHELL ["/bin/bash", "--login", "-c"]

# Initialize conda for the shell session
RUN conda init bash

# Create the conda environment and install required packages
RUN conda create --name tortoise python=3.9 numba inflect -y && \
    bash -c "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate tortoise && \
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    conda install transformers=4.31.0 scipy -y"

# Set conda environment to be activated by default in future RUN instructions
RUN echo "conda activate tortoise" >> ~/.bashrc

FROM conda_base AS runner

# Install the application
WORKDIR /app
RUN bash -c "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate tortoise && python setup.py install"

# Install FastAPI and Uvicorn
RUN bash -c "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate tortoise && pip install fastapi uvicorn"

# Copy the FastAPI app
COPY app /app/api

# Default command to run the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
