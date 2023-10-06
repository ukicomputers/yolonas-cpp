// Written by Uglješa Lukešević (github.com/ukicomputers)

#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class YoloNAS
{
private:
    cv::dnn::Net net;
    cv::Size outShape;

    void runPostProccessing(std::vector<std::vector<cv::Mat>> &out);

    float scoreThresh;
    float iouThresh;
    int pad_value;

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels, selectedIDX;
    std::vector<std::string> detectLabels;

public:
    struct detInf
    {
        int x, y, cx, cy, score;
        std::string label;
    };

    YoloNAS(std::string netPath, bool cuda, std::vector<int> imgsz, float score, float iou, int centerPadding, std::vector<std::string> lbls);
    void predict(cv::Mat &img, bool applyOverlayOnImage = true);
    std::vector<detInf> result;
};