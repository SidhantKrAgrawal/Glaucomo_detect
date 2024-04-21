FROM --platform=linux/amd64 tensorflow/tensorflow:latest-gpu

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV TF_ENABLE_ONEDNN_OPTS=0

RUN adduser --system --group user
RUN apt-get update && apt-get install -y libgl1-mesa-glx

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user model1.py /opt/app
COPY --chown=user:user model2.py /opt/app
COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user task1.h5 /opt/app
COPY --chown=user:user task2.h5 /opt/app

ENTRYPOINT ["python", "inference.py"]

