// Written by Uglješa Lukešević (github.com/ukicomputers)

#include "ukicomputers/YoloNAS.hpp"

YoloNAS::YoloNAS(string netPath, string metadata, bool cuda, vector<string> lbls, float scoreThresh)
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
    readConfig(metadata);
    detectLabels = lbls;
    outShape = cv::Size(cfg.width, cfg.height);

    if (scoreThresh != -1.00)
        cfg.score = scoreThresh;
}

void YoloNAS::runPostProccessing(vector<vector<cv::Mat>> &out)
{
    // Extract scores and bounding boxes
    cv::Mat &rawScores = out[0][0], &bboxes = out[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});
    bboxes.convertTo(bboxes, CV_32S);

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        double score;
        cv::Point classID;

        rowScores = rawScores.row(i);
        cv::minMaxLoc(rowScores, 0, &score, 0, &classID);

        // Check if the maximum score is above the threshold
        if ((float)score < cfg.score)
            continue;

        // Extract the bounding box coordinates
        vector<int> unsizedBox{bboxes.at<int>(i, 0), bboxes.at<int>(i, 1), bboxes.at<int>(i, 2), bboxes.at<int>(i, 3)};

        // Store the results
        labels.push_back(classID.x);
        scores.push_back(score);
        boxes.push_back(cv::Rect(unsizedBox[0], unsizedBox[1], (unsizedBox[2] - unsizedBox[0]), (unsizedBox[3] - unsizedBox[1])));
    }

    // Apply non-maximum suppression to remove redundant detections
    cv::dnn::NMSBoxes(boxes, scores, cfg.score, cfg.iou, suppressedObjs);

    // Release allocated memory
    bboxes.release();
    rawScores.release();
    rowScores.release();
}

void YoloNAS::readConfig(string filePath)
{
    // Read metadata configuration from a file
    ifstream file(filePath);
    string line;
    int cl = 1;

    if (!file.is_open())
    {
        // Close program if cannot open metadata
        cerr << "Cannot open metadata!" << endl;
        exit(-1);
    }

    while (getline(file, line))
    {
        // Parse configuration parameters
        if (cl == 1)
            cfg.iou = stof(line);
        else if (cl == 2)
            cfg.score = stof(line);
        else if (cl == 3)
            cfg.width = stof(line);
        else if (cl == 4)
            cfg.height = stof(line);
        else if (cl == 5)
            cfg.std = (line != "n") ? stof(line) : 0;
        else if (cl == 6)
            cfg.dlmr = (line == "t");
        else if (cl == 7)
            cfg.brm = (line != "n") ? stof(line) : 0;
        else if (cl == 8)
            cfg.cp = (line != "n") ? stof(line) : 0;
        else if (cl == 9)
        {
            if (line != "n")
            {
                for (int i = 9; i < 9 + 6; i++)
                {
                    getline(file, line);
                    cfg.norm.push_back(stof(line));
                }
            }
        }

        cl++;
    }

    file.close();
}

vector<YoloNAS::detInf> YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage)
{
    // Resize the input image to match the model's output shape
    cv::resize(img, img, outShape, 0, 0, cv::INTER_LINEAR);

    cv::Mat imgInput;
    img.copyTo(imgInput);

    // Resize the image while preserving the aspect ratio
    if (cfg.dlmr)
    {
        float scaleFactorX = (float)outShape.width / (float)imgInput.cols;
        float scaleFactorY = (float)outShape.height / (float)imgInput.rows;
        float scaleFactor = min(scaleFactorX, scaleFactorY);
        int newWidth = (int)round(imgInput.cols * scaleFactor);
        int newHeight = (int)round(imgInput.rows * scaleFactor);
        cv::resize(imgInput, imgInput, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    }

    // Pad detection to the bottom right or center
    if (cfg.brm > 0)
    {
        int padWidth = outShape.width - imgInput.rows;
        int padHeight = outShape.height - imgInput.cols;

        try
        {
            cv::copyMakeBorder(imgInput, imgInput, 0, padHeight, 0, padWidth, cv::BORDER_CONSTANT, cv::Scalar(cfg.brm, cfg.brm, cfg.brm));
        }
        catch (cv::Exception ex)
        {
            cerr << "Metadata does not match with model properties!" << endl;
            exit(-1);
        }
    }

    // Pad detection to the center
    if (cfg.cp > 0)
    {
        int padHeight = outShape.width - imgInput.rows;
        int padWidth = outShape.height - imgInput.cols;
        int padLeft = padWidth / 2;
        int padTop = padHeight / 2;

        try
        {
            cv::copyMakeBorder(imgInput, imgInput, padTop, padHeight - padTop, padLeft, padWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(cfg.cp, cfg.cp, cfg.cp));
        }
        catch (cv::Exception ex)
        {
            cerr << "Metadata does not match with model properties!" << endl;
            exit(-1);
        }
    }

    // Standardize the image if needed
    if (cfg.std > 0)
        imgInput.convertTo(imgInput, CV_32F, 1 / cfg.std);

    // Normalize the image if needed
    if (cfg.norm.size() > 0)
        imgInput = (imgInput - cv::Scalar(cfg.norm[3], cfg.norm[4], cfg.norm[5])) / cv::Scalar(cfg.norm[0], cfg.norm[1], cfg.norm[2]);

    // Create a blob from the image
    cv::dnn::blobFromImage(imgInput, imgInput, 1.0, cv::Size(), cv::Scalar(), true, false);

    vector<vector<cv::Mat>> outDet;

    net.setInput(imgInput);
    net.forward(outDet, net.getUnconnectedOutLayersNames());
    imgInput.release();

    runPostProccessing(outDet);
    outDet.clear();

    vector<YoloNAS::detInf> result;

    // Return detections from result of NMS
    for (auto &a : suppressedObjs)
    {
        YoloNAS::detInf currentDet;

        currentDet.x = boxes[a].x;
        currentDet.y = boxes[a].y;
        currentDet.w = boxes[a].width;
        currentDet.h = boxes[a].height;
        currentDet.score = scores[a];
        currentDet.label = detectLabels[labels[a]];

        if (applyOverlayOnImage)
        {
            // Adjust the coordinates of the bounding box to the original image size
            cv::Rect box(currentDet.x, currentDet.y, currentDet.w, currentDet.h);
            cv::rectangle(img, box, cv::Scalar(139, 255, 14), 2);

            // Put text on detected objects to visually see what is detected
            string text = currentDet.label + " - " + to_string(int(currentDet.score * 100)) + "%";
            cv::putText(img, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(56, 56, 255), 2);
        }

        result.push_back(currentDet);
    }

    clearCache();
    return result;
}

void YoloNAS::clearCache()
{
    // Clear the result-ish containers
    labels.clear();
    scores.clear();
    boxes.clear();
    suppressedObjs.clear();
}