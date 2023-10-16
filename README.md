# yolonas-cpp
An CPP library for object detection with full processing using OpenCV DNN (for ONNX) on YOLO-NAS model.

## Introduction
As normal world of AI today becomed reality, we use it as much as we can. Every _nerd_ knows that today to run an AI model, or anything similar to that, you need intermediate processing power that many people cannot afford it. As we, humans, we always choose the easiest path to do something. That is similar here. If you search online for some machine vision thing, you will probably 99% of time get something written in **Python**. Currently, **Python** is the easiest path, but in inference time not the fastest. If you understand what Python **_Interpreter_** is, you will understand how is it slow to do some complex tasks. There comes **CPP**. It almost runs everything today under the hood, and it's incredibly **FAST**. But **ONLY KNOWLEDGABLE PEOPLE** knows how to use it. So, here comes this libary. You can in **TWO LINES** of code run machine detection. It runs using **OpenCV**, and it can use **YOLO-NAS**, that model what is really good, fastest detection (inference) time, and it CAN detect extremly small objects. Here we can see how good it's efficency is:<br><br>
![efficency](https://github.com/ukicomputers/yolonas-cpp/assets/84191191/3c991abb-e1ed-49da-9cc0-0c37fcab7fe8)

## Detection
![detected](https://github.com/ukicomputers/yolonas-cpp/assets/84191191/800d2aa9-e564-4cd5-a8c8-a38328711fbc)
<br><br>Library uses **OpenCV** and it's **DNN** to run the model. Model is under **ONNX** format. It's also able to run on **CUDA**.

## Requirements
- GCC
- CMake
- OpenCV (installed as CMake library)
- CUDA compatible GPU + `nvcc` (_optionally, if you want to use GPU inference_)

## Setup
To install this library on your computer, firstly clone this repository:
```console
git clone https://github.com/ukicomputers/yolonas-cpp.git
```
Then change directory to it, and execute **install.sh** (_will require later sudo permission_), installing as CMake libary:
```console
cd yolonas-cpp
sh install.sh
```
After that library will be installed!

## Quick usage
**CPP** code (_not full code, just minimal example_):
```cpp
// include
#include <ukicomputers/YoloNAS.hpp>
//              modelpath           metadata   CUDA   Labels
YoloNAS net("./yolo_nas_s.onnx", "./metadata", false, {"lbl1", "lbl2"});
//          img  overlay (visually displayed detection)
net.predict(img, true);
```
Include library in **CMake**:
```cmake
find_package(YoloNAS REQUIRED)
target_link_libraries(${PROJECT_NAME} ukicomputers::YoloNAS)
```
Also required to include **OpenCV**:
```cmake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
```

## Full library usage
YoloNAS class argument requirements:
```cpp
YoloNAS::YoloNAS(string netPath, string metadata, bool cuda, vector<string> lbls)
```
**Initializes YoloNAS class.**
- modelpath (`std::string`),
- Metadata file (`std::string`),
- CUDA support (`bool`),
- labels in a vector (`std::vector<std::string>`)
```cpp
YoloNAS net(modelPath, metadata, CUDA, labels);
```
Void `predict`:
```cpp
void YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage)
```
**Predicts (detects) objects from image.**
- image (`cv::Mat`)
- visualy display detection (`bool`, writing on given image)
```cpp
net.predict(image, overlayOnImage);
```

Void `clearResults`:
```cpp
void YoloNAS::clearResults()
```
**Clears the results, only use if running real time.**
```cpp
net.clearResults();
```

## Demo
Demo is located in folder `demo` from downloaded repository. It will find `yolo_nas_s.onnx` model file from directory where it's executed from, and image `image.jpg` and use it for detection. To compile and run it, execute `build.sh` from `demo` folder.

## Custom model & metadata
To convert and use your own model, run it also inside library, [please follow this Colab notebook](https://colab.research.google.com/github/ukicomputers/yolonas-cpp/blob/main/notebooks/metadata.ipynb), or better, run it inside your machine by using `metadata.py`, [link here](https://github.com/ukicomputers/yolonas-cpp/blob/main/metadata.py). In `metadata.py`, first few variables you need to change according to your model (model path, model type, number of classes). Everything is nicely explained. **IMPORTANT: `metadata.py` DOES NOT ACCEPT `.onnx` FILE FORMAT!** It only accepts the standard YOLO `.pt` format.

## TODO
- normalize image
- read metadata in CPP without converting
- make detection visualisation look cooler

## License & contributions
Please add my name to top of a document to use library. It helps me feeling lot better!
```cpp
// YOLO-NAS Library written by Uglješa Lukešević (github.com/ukicomputers)
```
Also, feel free to open new issues, and contribute it with opening pull request!

## References
- [https://github.com/Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients)
