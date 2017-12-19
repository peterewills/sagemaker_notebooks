# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# base ubuntu container

FROM ubuntu:16.04

LABEL maintainer="v-peter.wills@thetradedesk.com"

# get programs needed

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# pip install python packages. link numpy and scipy to save space & time,
# clear cache to save space.

RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip install numpy scipy scikit-learn pandas flask gevent gunicorn && \
        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

# Set some environment variables.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# move directory 'algorithm' into the container
COPY algorithm /opt/program
WORKDIR /opt/program