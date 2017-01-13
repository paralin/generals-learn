#!/bin/bash

pushd image
docker build -t "paralin/tensorflow-keras:latest" .
popd
