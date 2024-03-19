// Written by Uglješa Lukešević (github.com/ukicomputers)

#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class YoloNAS
{
private:
    struct metadataConfig
    {
        float iou, score;
        int width, height;
        float std;
        bool dlmr;
        int brm, cp;
        vector<int> norm;
    };

    cv::dnn::Net net;
    cv::Size outShape;

    metadataConfig cfg;
    vector<string> labels;

    cv::Mat runPreProcessing(cv::Mat img);
    void runPostProccessing(vector<vector<cv::Mat>> out);
    void readConfig(string filePath);

    // Temporary vectors & variables
    vector<float> scores;
    vector<cv::Rect> boxes;
    vector<int> labelsID, suppressedObjs;

public:
    struct detInf
    {
        int x, y, w, h;
        float score;
        string label;
    };

    YoloNAS(string netPath, string config, bool cuda, vector<string> lbls, float scoreThresh = -1.00);
    vector<detInf> predict(cv::Mat img, bool applyOverlayOnImage = true);
};