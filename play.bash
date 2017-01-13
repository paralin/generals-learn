#!/bin/bash

sudo xhost +
docker run --rm -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --privileged \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $(pwd):/notebooks paralin/tensorflow-keras:latest python bot_neural_2.py
