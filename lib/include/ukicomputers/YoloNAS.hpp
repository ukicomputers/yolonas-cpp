// YOLO-NAS CPP library written by Uglješa Lukešević (github.com/ukicomputers)
// Marked as Open Source project under GNU GPL-3.0 license

#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class YoloNAS
{
public:
    struct detectionInfo
    {
        int x, y, w, h;
        float score;
        string label;
    };

    YoloNAS(string netPath, string config, vector<string> lbls, bool cuda = false);
    vector<detectionInfo> predict(cv::Mat &img, bool applyOverlayOnImage = true, float scoreThresh = -1.00);

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

    void readConfig(string filePath);
    cv::Mat runPreProcessing(cv::Mat &img);
    void exceptionHandler(int ex);
    void painter(cv::Mat &img, detectionInfo &detection);

    // Inputs are defined as destinstions, like cv:: call, except our own implementation
    void runPostProccessing(vector<vector<cv::Mat>> &input,
                            vector<cv::Rect> &boxesOut,
                            vector<int> &labelsOut,
                            vector<float> &scoresOut,
                            vector<int> &suppressedObjs,
                            float &scoreThresh);
};