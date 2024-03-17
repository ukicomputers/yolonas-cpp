#!/bin/bash

git submodule init
git submodule update --recursive --force --remote

echo -e "\nModels sucessfully downloaded! \n"
