// Written by Uglješa Lukešević (github.com/ukicomputers)

#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class YoloNAS
{
private:
    cv::dnn::Net net;
    cv::Size outShape;

    void runPostProccessing(vector<vector<cv::Mat>> &out);

    vector<float> scores;
    vector<cv::Rect> boxes;
    vector<int> labels, selectedIDX;
    vector<string> detectLabels;

    struct metadataConfig
    {
        float iou, score;
        int width, height;
        float std;
        bool dr, dlmr;
        int brm, cp;
    };

    vector<metadataConfig> readConfig(string filePath);
    vector<metadataConfig> cfg;

public:
    struct detInf
    {
        int x, y, cx, cy, score;
        string label;
    };

    YoloNAS(string netPath, string config, bool cuda, vector<string> lbls);
    void predict(cv::Mat &img, bool applyOverlayOnImage = true);
    void clearResults();
    vector<detInf> result;    
};