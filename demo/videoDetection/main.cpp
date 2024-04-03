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

int main()
{
    // Initialize FPS counter
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

    // Make an capture (currently from camera source, you can also use and video, just specify it's path in string)
    cv::VideoCapture cap(0);

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

        // Run the FPS counter
        begin = chrono::steady_clock::now();

        // Simply run net.predict(frame) to detect with overlay
        net.predict(frame, true, score);

        // Stop the FPS counter and show the count
        end = chrono::steady_clock::now();
        float fps = 1000.0 / float(chrono::duration_cast<chrono::microseconds>(end - begin).count());
        fps = roundf(fps * 100) / 100;
        cv::putText(frame, "FPS: " + to_string(fps), cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0));

        // Show the result
        cv::imshow("detection", frame);

        // If pressed ESC, close the program
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    // Release the capture
    cap.release();
    return 0;
}