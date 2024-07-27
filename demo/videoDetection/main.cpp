// Written by Uglješa Lukešević (github.com/ukicomputers)

#include <ukicomputers/YoloNAS.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;

// This is vector for already trained (by deci.ai) YOLO-NAS COCO dataset
const vector<string> COCO_LABELS{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                 "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                                 "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                 "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                 "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                 "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                                 "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                                 "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// Head directory of all models
const string modelsPath = "../../../models/yolonas/onnx/";

// Used re-defined score thereshold
float score = 0.5;

// Write video
bool writeVideo = true;

int main()
{
    // Initialize time counter
    chrono::steady_clock::time_point begin;
    chrono::steady_clock::time_point end;

    /*  YoloNAS class argument requirements:
            modelpath (std::string),
            metadata path (std::string),
            labels in a vector (std::vector<std::string>)
            CUDA support (bool),

        All of this information you can find in Python script that gets info
        from selected YOLO-NAS model. Script is in repo.
    */

    // Prepare YoloNAS
    YoloNAS net(modelsPath + "yolonas_s.onnx", modelsPath + "yolonas_s_metadata", COCO_LABELS, false);
    net.warmupModel();

    // Make an capture (currently from file, you can also use and camera source, just insert it's ID)
    cv::VideoCapture cap(modelsPath + "street.mp4");
    
    // Make VideoWriter if bool is true
    cv::VideoWriter video;
    if(writeVideo)
        video.open("detection.avi", cv::VideoWriter::fourcc('M','J','P','G'), 24, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    // If we cannot open capture, we are returning an error
    if (!cap.isOpened())
    {
        cerr << "capture not open" << endl;
        return (-1);
    }

    while (1)
    {
        // Get the current image
        cv::Mat frame;
        cap >> frame;

        // If it's last frame
        if (frame.empty())
            break;

        /* Argument requirements for net.predict void:
            cv::Mat (image),
            bool overlayOnImage = true (for visually representative detection),
            int scoreThreshold = -1.0 (if passed, detector will use passed thereshold, otherwise, model default)
        */

        // Run the time counter
        begin = chrono::steady_clock::now();

        // Simply run net.predict(frame) to detect with overlay
        net.predict(frame, true, score);

        // Stop the time counter and show the count
        end = chrono::steady_clock::now();
        int inference = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
        cv::putText(frame, "Inference time: " + to_string(inference) + "ms", cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(255, 255, 0));

        // Show the result
        cv::imshow("detection", frame);

        // Write video if possible
        if(writeVideo)
            video.write(frame);

        // If pressed ESC, close the program
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    // Release the capture and video
    cap.release();
    video.release();

    return 0;
}