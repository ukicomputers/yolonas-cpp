// YOLO-NAS CPP library written by Uglješa Lukešević (github.com/ukicomputers)
// Marked as Open Source project under GNU GPL-3.0 license

#include "ukicomputers/YoloNAS.hpp"

YoloNAS::YoloNAS(string netPath, string config, vector<string> lbls, bool cuda)
{
    // Load the neural network model from an ONNX file
    try
    {
        net = cv::dnn::readNetFromONNX(netPath);
    }
    catch (cv::Exception ex)
    {
        exceptionHandler(0);
    }

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
    readConfig(config);
    labels = lbls;
    outShape = cv::Size(cfg.width, cfg.height);

    // Load model into memory
    warmupModel();
}

void YoloNAS::warmupModel()
{
    cv::Mat input({1, 3, cfg.width, cfg.height}, CV_32F);
    cv::randu(input, cv::Scalar(0), cv::Scalar(1)); // fill matrix
    net.setInput(input);
    net.forward();
    input.release();
}

cv::Mat YoloNAS::runPreProcessing(cv::Mat &img)
{
    cv::Mat imgInput;

    // Resize the image while preserving the aspect ratio
    if (cfg.dlmr)
    {
        // Applying scale factors for only for expected MODEL input
        float scaleX = (float)outShape.width / (float)img.cols;
        float scaleY = (float)outShape.height / (float)img.rows;
        int newWidth = round(img.cols * scaleX);
        int newHeight = round(img.rows * scaleY);
        cv::resize(img, imgInput, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR); // copying img here or down
    }
    else
    {
        // Resize the input image to match the model's output shape
        cv::resize(img, imgInput, outShape, 0, 0, cv::INTER_LINEAR);
    }

    // Pad sizes
    int padHeight = outShape.width - imgInput.rows;
    int padWidth = outShape.height - imgInput.cols;

    // Pad detection to the bottom right or center
    if (cfg.brm > 0)
    {
        try
        {
            cv::copyMakeBorder(imgInput, imgInput, 0, padHeight, 0, padWidth, cv::BORDER_CONSTANT, cv::Scalar(cfg.brm, cfg.brm, cfg.brm));
        }
        catch (cv::Exception ex)
        {
            exceptionHandler(1);
        }
    }

    // Pad detection to the center
    if (cfg.cp > 0)
    {
        int padLeft = padWidth / 2;
        int padTop = padHeight / 2;

        try
        {
            cv::copyMakeBorder(imgInput, imgInput, padTop, padHeight - padTop, padLeft, padWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(cfg.cp, cfg.cp, cfg.cp));
        }
        catch (cv::Exception ex)
        {
            exceptionHandler(1);
        }
    }

    // Normalize the image if needed
    if (cfg.norm.size() > 0)
        imgInput = (imgInput - cv::Scalar(cfg.norm[3], cfg.norm[4], cfg.norm[5])) / cv::Scalar(cfg.norm[0], cfg.norm[1], cfg.norm[2]);

    // Standardize the image if needed
    if (cfg.std > 0)
        imgInput.convertTo(imgInput, CV_32F, 1 / cfg.std);

    // Create a blob from the image
    cv::dnn::blobFromImage(imgInput, imgInput, 1.0, cv::Size(), cv::Scalar(), true, false);

    return imgInput;
}

void YoloNAS::runPostProccessing(vector<vector<cv::Mat>> &input,
                                 vector<cv::Rect> &boxesOut,
                                 vector<int> &labelsOut,
                                 vector<float> &scoresOut,
                                 vector<int> &suppressedObjs,
                                 float &scoreThresh)
{
    // Extract scores and bounding boxes
    cv::Mat rawScores = input[0][0], bboxes = input[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});
    bboxes.convertTo(bboxes, CV_32S); // convert coordinates to ints

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        double score;
        cv::Point classID;

        rowScores = rawScores.row(i);
        cv::minMaxLoc(rowScores, 0, &score, 0, &classID);

        // Check if the maximum score is above the threshold
        if ((float)score < scoreThresh)
            continue;

        // Extract the bounding box coordinates
        vector<int> unsizedBox{bboxes.at<int>(i, 0), bboxes.at<int>(i, 1), bboxes.at<int>(i, 2), bboxes.at<int>(i, 3)};

        // Store the results
        labelsOut.push_back(classID.x);
        scoresOut.push_back(score);
        boxesOut.push_back(cv::Rect(unsizedBox[0], unsizedBox[1], (unsizedBox[2] - unsizedBox[0]), (unsizedBox[3] - unsizedBox[1])));
    }

    // Apply non-maximum suppression to remove redundant detections
    cv::dnn::NMSBoxes(boxesOut, scoresOut, scoreThresh, cfg.iou, suppressedObjs);

    // Release allocated memory
    bboxes.release();
    rawScores.release();
    rowScores.release();
}

void YoloNAS::readConfig(string filePath)
{
    ifstream file(filePath);
    string line;
    int cl = 1;

    if (!file.is_open())
    {
        exceptionHandler(2);
    }

    while (getline(file, line))
    {
        switch (cl)
        {
        case 1:
            cfg.iou = stof(line);
            break;
        case 2:
            cfg.score = stof(line);
            break;
        case 3:
            cfg.width = stof(line);
            break;
        case 4:
            cfg.height = stof(line);
            break;
        case 5:
            cfg.std = (line != "n") ? stof(line) : 0;
            break;
        case 6:
            cfg.dlmr = (line == "t");
            break;
        case 7:
            cfg.brm = (line != "n") ? stof(line) : 0;
            break;
        case 8:
            cfg.cp = (line != "n") ? stof(line) : 0;
            break;
        case 9:
            if (line != "n")
            {
                for (int i = 9; i < 9 + 6; i++)
                {
                    getline(file, line);
                    cfg.norm.push_back(stof(line));
                }
            }
            break;
        }
        cl++;
    }

    file.close();
}

void YoloNAS::painter(cv::Mat &img, YoloNAS::detectionInfo &detection)
{
    // Adjust the coordinates of the bounding box to the original image size
    cv::Rect box(detection.x, detection.y, detection.w, detection.h);
    cv::rectangle(img, box, cv::Scalar(139, 255, 14), 2);

    // Put text on detected objects to visually see what is detected
    string text = detection.label + " - " + to_string(int(detection.score * 100)) + "%";
    cv::putText(img, text, cv::Point(detection.x, detection.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(56, 56, 255), 2);
}

void YoloNAS::exceptionHandler(int ex)
{
    // Throw the called exception
    switch (ex)
    {
    case 0:
        throw runtime_error("MODEL_LOADING_FAILED");
    case 1:
        throw runtime_error("METADATA_MISMATCHES_MODEL");
    case 2:
        throw runtime_error("METADATA_NOT_FOUND");
    }
}

vector<YoloNAS::detectionInfo> YoloNAS::predict(cv::Mat &img, bool applyOverlayOnImage, float scoreThresh)
{
    vector<vector<cv::Mat>> outDet;
    cv::Mat processedImg = runPreProcessing(img); // Preprocess the image

    // Get raw results from inference
    net.setInput(processedImg);
    net.forward(outDet, net.getUnconnectedOutLayersNames());

    // Free memory
    processedImg.release();

    // Get score thresh
    if (scoreThresh < 0)
        scoreThresh = cfg.score;

    // Required vectors for post processer
    vector<float> scores;
    vector<cv::Rect> boxes;
    vector<int> detectionLabels, suppressedObjs;

    // Run result processing
    runPostProccessing(outDet, boxes, detectionLabels, scores, suppressedObjs, scoreThresh);

    // Free memory
    outDet.clear();

    // Returned vector
    vector<YoloNAS::detectionInfo> result;

    // Applying scale factors for only for IMAGE (from already preprocessed steps)
    float scaleX = (float)img.cols / (float)outShape.width;
    float scaleY = (float)img.rows / (float)outShape.height;

    // Return detections from result of NMS
    for (auto i : suppressedObjs)
    {
        YoloNAS::detectionInfo currentDet;

        // Adjust bounding box coordinates to original image size using scaling factors
        currentDet.x = int(boxes[i].x * scaleX);
        currentDet.y = int(boxes[i].y * scaleY);
        currentDet.w = int(boxes[i].width * scaleX);
        currentDet.h = int(boxes[i].height * scaleY);
        currentDet.score = scores[i];
        currentDet.label = labels[detectionLabels[i]];

        if (applyOverlayOnImage)
        {
            painter(img, currentDet);
        }

        result.push_back(currentDet);
    }

    // Free memory
    detectionLabels.clear();
    scores.clear();
    boxes.clear();
    suppressedObjs.clear();

    return result;
}