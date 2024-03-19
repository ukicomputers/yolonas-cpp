#!/bin/bash

echo -e '\nDo not forget models while running inference! \nYou can download models by executing download_models.sh in home dir of this repo.\n'

rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

echo -e "\n"

./yolonas-demo