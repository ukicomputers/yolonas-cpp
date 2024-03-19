#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

echo -e '\nDo not forget models while running inference! \nYou can download models by executing download_models.sh in home dir of this repo.\nMake sure that YOU ARE in build folder when running demos, otherwise, program will not find models source dir.\n'