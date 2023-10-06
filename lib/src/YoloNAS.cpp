// Written by Uglješa Lukešević (github.com/ukicomputers)

#include "ukicomputers/YoloNAS.hpp"

YoloNAS::YoloNAS(std::string netPath, bool cuda, std::vector<int> imgsz, float score, float iou, int centerPadding, std::vector<std::string> lbls)
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

    scoreThresh = score;
    iouThresh = iou;
    pad_value = centerPadding;
    detectLabels = lbls;

    outShape = cv::Size(imgsz[0], imgsz[1]);
}

void YoloNAS::runPostProccessing(std::vector<std::vector<cv::Mat>> &out)
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

        if ((float)maxScore < scoreThresh)
            continue;

        std::vector<float> box{bboxes.at<float>(i, 0),  // x
                               bboxes.at<float>(i, 1),  // y
                               bboxes.at<float>(i, 2),  // cx
                               bboxes.at<float>(i, 3)}; // cy

        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect((int)box[0], (int)box[1], (int)(box[2] - box[0]), (int)(box[3] - box[1])));
    }

    cv::dnn::NMSBoxes(boxes, scores, scoreThresh, iouThresh, selectedIDX);

    bboxes.release();
    rawScores.release();
    rowScores.release();
}

void YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage)
{
    cv::Mat imgInput;
    img.copyTo(imgInput);

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

    // Calculate padding to center the image in the output shape
    int padWidth = outShape.width - newWidth;
    int padHeight = outShape.height - newHeight;
    int padLeft = padWidth / 2;
    int padTop = padHeight / 2;

    // Pad the image to fit the output shape
    cv::copyMakeBorder(imgInput, imgInput, padTop, padHeight - padTop, padLeft, padWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));

    // Standardize the image
    imgInput.convertTo(imgInput, CV_32F, 1 / 255.0);

    // Create a blob from the image
    cv::dnn::blobFromImage(imgInput, imgInput, 1.0, cv::Size(), cv::Scalar(), true, false);

    std::vector<std::vector<cv::Mat>> outDet;

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
        std::string label = detectLabels[labels[a]];

        if (applyOverlayOnImage)
        {
            // Adjust the coordinates of the bounding box to the original image size
            cv::Rect box(x, y, cx, cy);
            cv::rectangle(img, box, cv::Scalar(139, 255, 14), 2);

            // Put text on detected objects to visually see what is detected
            std::string text = label + " - " + std::to_string(score) + "%";
            cv::putText(img, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(56, 56, 255), 2);
        }

        YoloNAS::result.push_back({x, y, cx, cy, score, label});
    }
}
