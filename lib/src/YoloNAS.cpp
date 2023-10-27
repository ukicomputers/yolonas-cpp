// Written by Uglješa Lukešević (github.com/ukicomputers)

#include "ukicomputers/YoloNAS.hpp"

YoloNAS::YoloNAS(string netPath, string metadata, bool cuda, vector<string> lbls)
{
    // Load the neural network model from an ONNX file
    net = cv::dnn::readNetFromONNX(netPath);

    // Set the preferable backend and target based on CUDA availability
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

    // Read and store configuration settings
    cfg = readConfig(metadata);
    detectLabels = lbls;
    outShape = cv::Size(cfg[0].width, cfg[0].height);
}

void YoloNAS::runPostProccessing(vector<vector<cv::Mat>> &out)
{
    // Extract scores and bounding boxes
    cv::Mat &rawScores = out[0][0], &bboxes = out[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        rowScores = rawScores.row(i);
        cv::Point classID;
        double maxScore;
        cv::minMaxLoc(rowScores, 0, &maxScore, 0, &classID);

        // Check if the maximum score is above the threshold
        if ((float)maxScore < cfg[0].score)
            continue;

        // Extract the bounding box coordinates
        vector<float> box{bboxes.at<float>(i, 0), bboxes.at<float>(i, 1),
                          bboxes.at<float>(i, 2), bboxes.at<float>(i, 3)};

        // Store the results
        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect((int)box[0], (int)box[1], (int)(box[2] - box[0]), (int)(box[3] - box[1])));
    }

    // Apply non-maximum suppression to remove redundant detections
    cv::dnn::NMSBoxes(boxes, scores, cfg[0].score, cfg[0].iou, selectedIDX);

    // Release allocated memory
    bboxes.release();
    rawScores.release();
    rowScores.release();
}

vector<YoloNAS::metadataConfig> YoloNAS::readConfig(string filePath)
{
    // Read metadata configuration from a file
    ifstream file(filePath);
    string line;
    int cl = 1;

    float iou, score;
    int width, height;
    float std;
    bool dlmr;
    int brm, cp;

    if (!file.is_open())
    {
        // Close program if cannot open metadata
        cerr << "cannot open metadata!" << endl;
        exit(-1);
    }

    while (getline(file, line))
    {
        // Parse configuration parameters
        if (cl == 1)
            iou = stof(line);
        else if (cl == 2)
            score = stof(line);
        else if (cl == 3)
            width = stof(line);
        else if (cl == 4)
            height = stof(line);
        else if (cl == 5)
            std = (line != "n") ? stof(line) : 0;
        else if (cl == 6)
            dlmr = (line == "t");
        else if (cl == 7)
            brm = (line != "n") ? stof(line) : 0;
        else if (cl == 8)
            cp = (line != "n") ? stof(line) : 0;
        cl++;
    }

    file.close();
    vector<metadataConfig> tmp;
    tmp.push_back({iou, score, width, height, std, dlmr, brm, cp});
    return tmp;
}

void YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage)
{
    // Resize the input image to match the model's output shape
    cv::resize(img, img, outShape, 0, 0, cv::INTER_LINEAR);
    
    cv::Mat imgInput;
    img.copyTo(imgInput);

    // Resize the image while preserving the aspect ratio
    if (cfg[0].dlmr)
    {
        float scaleFactorX = (float)outShape.width / (float)imgInput.cols;
        float scaleFactorY = (float)outShape.height / (float)imgInput.rows;
        float scaleFactor = std::min(scaleFactorX, scaleFactorY);
        int newWidth = (int)std::round(imgInput.cols * scaleFactor);
        int newHeight = (int)std::round(imgInput.rows * scaleFactor);
        cv::resize(imgInput, imgInput, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    }

    // Pad detection to the bottom right or center
    if (cfg[0].brm > 0)
    {
        int padWidth = outShape.width - imgInput.rows;
        int padHeight = outShape.height - imgInput.cols;

        try
        {
            cv::copyMakeBorder(imgInput, imgInput, 0, padHeight, 0, padWidth, cv::BORDER_CONSTANT, cv::Scalar(cfg[0].brm, cfg[0].brm, cfg[0].brm));
        }
        catch (cv::Exception ex)
        {
            cerr << "Metadata does not match with model properties!" << endl;
            exit(-1);
        }
    }

    // Pad detection to the center
    int padLeft, padTop = 0;
    if (cfg[0].cp > 0)
    {
        int padHeight = outShape.width - imgInput.rows;
        int padWidth = outShape.height - imgInput.cols;
        padLeft = padWidth / 2;
        padTop = padHeight / 2;

        try
        {
            cv::copyMakeBorder(imgInput, imgInput, padTop, padHeight - padTop, padLeft, padWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(cfg[0].cp, cfg[0].cp, cfg[0].cp));
        }
        catch (cv::Exception ex)
        {
            cerr << "Metadata does not match with model properties!" << endl;
            exit(-1);
        }
    }

    // Standardize the image if needed
    if (cfg[0].std > 0)
        imgInput.convertTo(imgInput, CV_32F, 1 / cfg[0].std);

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

void YoloNAS::clearResults()
{
    // Clear the result containers
    labels.clear();
    scores.clear();
    boxes.clear();
    result.clear();
    selectedIDX.clear();
}