// Written by Uglješa Lukešević (github.com/ukicomputers)

#include <ukicomputers/YoloNAS.hpp>
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

    // Make an capture (currently from video file)
    cv::VideoCapture cap("./video.mp4");

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

        // Simply run net.predict(frame) to detect with overlay
        net.predict(frame);

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