// Written by Uglješa Lukešević (github.com/ukicomputers)

#include "ukicomputers/YoloNAS.hpp"

YoloNAS::YoloNAS(string netPath, string metadata, bool cuda, vector<string> lbls)
{
    net = cv::dnn::readNetFromONNX(netPath);
    if (cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    cfg = readConfig(metadata);
    detectLabels = lbls;
    outShape = cv::Size(cfg[0].width, cfg[0].height);
}

void YoloNAS::runPostProccessing(vector<vector<cv::Mat>> &out)
{
    // Gets coordinates, scores and labels. Then runs NMSBoxes to delete wrong mulitple detections.
    cv::Mat &rawScores = out[0][0], &bboxes = out[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        rowScores = rawScores.row(i);
        cv::Point classID;
        double maxScore;
        minMaxLoc(rowScores, 0, &maxScore, 0, &classID);

        if ((float)maxScore < cfg[0].score)
            continue;

        vector<float> box{bboxes.at<float>(i, 0),  // x
                          bboxes.at<float>(i, 1),  // y
                          bboxes.at<float>(i, 2),  // width
                          bboxes.at<float>(i, 3)}; // height

        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect((int)box[0], (int)box[1], (int)(box[2] - box[0]), (int)(box[3] - box[1])));
    }

    cv::dnn::NMSBoxes(boxes, scores, cfg[0].score, cfg[0].iou, selectedIDX);

    bboxes.release();
    rawScores.release();
    rowScores.release();
}

vector<YoloNAS::metadataConfig> YoloNAS::readConfig(string filePath)
{
    // Metadata reader, nothing special, don't look too much, you may be confused
    ifstream file(filePath);
    string line;
    int cl = 1;

    float iou, score;
    int width, height;
    float std;
    bool dr, dlmr;
    int brm, cp;

    while (getline(file, line))
    {
        if (cl == 1)
        {
            iou = stof(line);
        }
        else if (cl == 2)
        {
            score = stof(line);
        }
        else if (cl == 3)
        {
            width = stof(line);
        }
        else if (cl == 4)
        {
            height = stof(line);
        }
        else if (cl == 5)
        {
            if (line != "n")
            {
                std = stof(line);
            }
        }
        else if (cl == 6)
        {
            if (line == "t")
                dr = true;
            else
                dr = false;
        }
        else if (cl == 7)
        {
            if (line == "t")
                dlmr = true;
            else
                dlmr = false;
        }
        else if (cl == 8)
        {
            if (line != "n")
            {
                brm = stof(line);
            }
        }
        else if (cl == 9)
        {
            if (line != "n")
            {
                cp = stof(line);
            }
        }
        cl++;
    }

    file.close();
    vector<metadataConfig> tmp;
    tmp.push_back({iou, score, width, height, std, dr, dlmr, brm, cp});
    return tmp;
}

void YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage)
{
    cv::Mat imgInput;
    img.copyTo(imgInput);

    // Standardize the image
    if (cfg[0].std > 0)
        imgInput.convertTo(imgInput, CV_32F, 1 / cfg[0].std);

    // Resize to model picture proportion
    if (cfg[0].dr == true)
    {
        cv::resize(img, img, outShape, 0, 0, cv::INTER_LINEAR);
        cv::resize(imgInput, imgInput, outShape, 0, 0, cv::INTER_LINEAR);
    }

    // Resize the image while preserving the aspect ratio
    if (cfg[0].dlmr == true)
    {
        // Calculate the scale factor for resizing
        float scaleFactorX = (float)outShape.width / (float)imgInput.cols;
        float scaleFactorY = (float)outShape.height / (float)imgInput.rows;

        // Choose the minimum scale factor to fit the image within the output shape
        float scaleFactor = std::min(scaleFactorX, scaleFactorY);

        // Calculate the new dimensions after resizing
        int newWidth = (int)std::round(imgInput.cols * scaleFactor);
        int newHeight = (int)std::round(imgInput.rows * scaleFactor);

        // Resize the image while preserving the aspect ratio
        cv::resize(img, img, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
        cv::resize(imgInput, imgInput, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    }

    // Pad detection to bottom right
    if (cfg[0].brm > 0)
    {
        // Calculate padding to center the image in the output shape
        int padWidth = outShape.width - imgInput.rows;
        int padHeight = outShape.height - imgInput.cols;

        // Pad the image to fit the output shape
        cv::copyMakeBorder(imgInput, imgInput, 0, padHeight, 0, padWidth, cv::BORDER_CONSTANT, cv::Scalar(cfg[0].brm, cfg[0].brm, cfg[0].brm));
    }

    // Define if we change coordinates
    int padLeft, padTop;

    // Pad detection to center
    if (cfg[0].cp > 0)
    {
        // Calculate padding to center the image in the output shape
        int padHeight = outShape.width - imgInput.rows;
        int padWidth = outShape.height - imgInput.cols;
        padLeft = padWidth / 2;
        padTop = padHeight / 2;

        // Pad the image to fit the output shape
        cv::copyMakeBorder(imgInput, imgInput, padTop, padHeight - padTop, padLeft, padWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(cfg[0].cp, cfg[0].cp, cfg[0].cp));
    }

    // Create a blob from the image
    cv::dnn::blobFromImage(imgInput, imgInput, 1.0, cv::Size(), cv::Scalar(), true, false);

    vector<vector<cv::Mat>> outDet;

    net.setInput(imgInput);
    net.forward(outDet, net.getUnconnectedOutLayersNames());

    imgInput.release();

    runPostProccessing(outDet);

    for (auto &a : selectedIDX)
    {
        int x = boxes[a].x - padLeft;
        int y = boxes[a].y - padTop;
        int cx = boxes[a].width;
        int cy = boxes[a].height;
        int score = scores[a] * 100;
        string label = detectLabels[labels[a]];

        if (applyOverlayOnImage)
        {
            // Adjust the coordinates of the bounding box to the original image size
            cv::Rect box(x, y, cx, cy);
            cv::rectangle(img, box, cv::Scalar(139, 255, 14), 2);

            // Put text on detected objects to visually see what is detected
            string text = label + " - " + to_string(score) + "%";
            cv::putText(img, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(56, 56, 255), 2);
        }

        result.push_back({x, y, cx, cy, score, label});
    }
}