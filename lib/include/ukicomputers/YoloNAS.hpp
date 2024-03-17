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
    vector<int> labels, suppressedObjs;
    vector<string> detectLabels;

    struct metadataConfig
    {
        float iou, score;
        int width, height;
        float std;
        bool dlmr;
        int brm, cp;
        vector<int> norm;
    };

    metadataConfig cfg;
    void readConfig(string filePath);
    void clearCache();

public:
    struct detInf
    {
        int x, y, w, h, score;
        string label;
    };

    YoloNAS(string netPath, string config, bool cuda, vector<string> lbls, float scoreThresh = -1.00);
    vector<detInf> predict(cv::Mat &img, bool applyOverlayOnImage = true);
};