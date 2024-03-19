#!/bin/bash

if [[ "$EUID" -ne 0 ]]; then
    echo "To install library, this script needs to be executed with superuser permissions."
    echo "Please run this script with sudo command (sudo bash install.bash)"
    exit -1
fi

cd lib
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
make install

echo -e "\nLibrary is sucessfully installed.\n"