#!/bin/bash

set +x
docker run --rm -it \
  -v $(pwd):/notebooks paralin/tensorflow-keras:latest \
  python replay_convert.py $1
