#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "linux" ]]; then
    echo "Using system boost package"
    exit
fi

# TODO: download and build libboost.python
