// Written by Uglješa Lukešević (github.com/ukicomputers)

#include <ukicomputers/YoloNAS.hpp>
#include <chrono>
#include <cmath>
#include <iostream>

// This is vector for already trained (by deci.ai) YOLO-NAS COCO dataset
const std::vector<std::string> COCO_LABELS{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                                           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                                           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

int main()
{
    // Initialize FPS counter
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    /*  YoloNAS class argument requirements:
            modelpath (std::string),
            metadata path (std::string),
            CUDA support (bool),
            labels in a vector (std::vector<std::string>)

        All of this information you can find in Python script that gets info
        from selected YOLO-NAS model. Script is in repo.
    */

    // Prepare YoloNAS
    YoloNAS net("./model.onnx", "./metadata", false, COCO_LABELS);

    // Make an capture (currently from camera source, you can also use and video, just specify it's path in string)
    cv::VideoCapture cap(0);

    // If we cannot open capture, we are returning an error
    if (!cap.isOpened())
    {
        std::cerr << "capture not open" << std::endl;
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

        // We need to clear results before again detecting
        net.clearResults();

        /* Argument requirements for net.predict void:
            cv::Mat (image),
            bool overlayOnImage = true (for visually representative detection)
        */

        // Run the FPS counter
        begin = std::chrono::steady_clock::now();

        // Simply run net.predict(frame) to detect with overlay
        net.predict(frame);

        // Stop the FPS counter and show the count
        end = std::chrono::steady_clock::now();
        float fps = 1000.0 / static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        fps = std::roundf(fps * 100) / 100;
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