FROM ubuntu:22.04

# Accept build arguments
ARG EPLUS_VERSION
ARG EPLUS_BUILD
ARG EPLUS_DIR
ARG EPLUS_TAR_FILENAME
ARG EPLUS_URL
ARG DEBIAN_FRONTEND

# install dependencies
RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-pip \
    tar \
    wget \
    git
# wget the energyplus tarball
RUN wget -q ${EPLUS_URL} -O ${EPLUS_TAR_FILENAME}
# Install EnergyPlus in our directory
RUN mkdir ${EPLUS_DIR}
RUN tar -xvzf ${EPLUS_TAR_FILENAME} -C ${EPLUS_DIR} --strip-components=1
# Set up environment for EnergyPlus
ENV PATH="${EPLUS_DIR}:${PATH}"
ENV LD_LIBRARY_PATH="${EPLUS_DIR}:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${EPLUS_DIR}:${PYTHONPATH}"
ENV ENERGYPLUS_CMD="${EPLUS_DIR}/energyplus"
# test that energyplus exists
RUN echo energyplus --version
# Install uv
RUN pip install uv
# Install packages
WORKDIR /code
EXPOSE 8000
#RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/supsi-dacd-isaac/dream.git /code
COPY . .
RUN uv sync