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

    // Prepare Image
    cv::Mat img = cv::imread("./image.jpg");

    /* Argument requirements for net.predict void:
        cv::Mat (image),
        bool overlayOnImage = true (for visually representative detection)
    */

    // Simply run net.predict(img) to detect with overlay
    net.predict(img);

    /* Defined vector for detection result:
        net.result[i].x - X coordinate of detected object (int)
        net.result[i].y - Y coordinate of detected object (int)
        net.result[i].cx - Width of detected object (int)
        net.result[i].cy - Height of detected object (int)
        net.result[i].label - Name of detetected object (std::string)
        net.result[i].score - Accuracy of detected object (float)

        (here int i is used as example for object detection sequence number)
    */

    for (int i = 0; i < net.result.size(); i++)
    {
        std::cout << "************" << std::endl;
        std::cout << "Detected: " + net.result[i].label << std::endl;
        std::cout << "Score: " + std::to_string(net.result[i].score) + " %" << std::endl;
        std::cout << "X, Y: " + std::to_string(net.result[i].x) + ", " + std::to_string(net.result[i].y) << std::endl;
        std::cout << "CX (width), CY (height): " + std::to_string(net.result[i].cx) + ", " + std::to_string(net.result[i].cy) << std::endl;
    }

    // Write & show image to file for showcase
    cv::imwrite("./detected.jpg", img);
    cv::imshow("detected", img);
    cv::waitKey(0);

    return 0;
}