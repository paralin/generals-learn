#!/bin/bash

docker run --rm -it -v $(pwd):/notebooks paralin/tensorflow-keras:latest python exp_learn.py
