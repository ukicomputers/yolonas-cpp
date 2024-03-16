#!/bin/bash

git submodule init
git submodule update --recursive --force --merge

echo -e "\nModels sucessfully downloaded! \n"
