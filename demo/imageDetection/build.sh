rm -rf build
mkdir build
cd build
cmake ..
make
echo "Don't forget model.onnx and metadata file while running program!"