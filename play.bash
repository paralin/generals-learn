#!/bin/bash

sudo xhost +
  # --restart=always \
docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --rm \
  --privileged \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $(pwd):/notebooks paralin/tensorflow-keras:latest python bot_neural.py
