// Written by Uglješa Lukešević (github.com/ukicomputers)

#include <ukicomputers/YoloNAS.hpp>
#include <iostream>
#include <chrono>
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

    // Prepare Image
    cv::Mat img = cv::imread(modelsPath + "image.jpg");

    /* Argument requirements for net.predict void:
        cv::Mat (image),
        bool overlayOnImage = true (for visually representative detection),
        int scoreThreshold = -1.0 (if passed, detector will use passed thereshold, otherwise, model default)
    */

    // Run the time counter
    begin = chrono::steady_clock::now();

    // Simply run net.predict(img) to detect with overlay
    auto result = net.predict(img, true, score);

    // Stop the time counter and show the count
    end = chrono::steady_clock::now();
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << endl << endl;

    /* Defined vector for detection result:
        result[i].x - X coordinate of detected object (int)
        result[i].y - Y coordinate of detected object (int)
        result[i].cx - Width of detected object (int)
        result[i].cy - Height of detected object (int)
        result[i].label - Name of detetected object (std::string)
        result[i].score - Accuracy of detected object (float)

        (here int i is used as example for object detection sequence number)
    */

    for (int i = 0; i < result.size(); i++)
    {
        cout << "************" << endl;
        cout << "Detected: " + result[i].label << endl;
        cout << "Score: " + to_string(result[i].score) << endl;
        cout << "X, Y: " + to_string(result[i].x) + ", " + to_string(result[i].y) << endl;
        cout << "W, H: " + to_string(result[i].w) + ", " + to_string(result[i].h) << endl;
    }

    // Write & show image to file for showcase
    cv::imwrite("./detected.jpg", img);
    cv::imshow("detected", img);
    cv::waitKey(0);

    return 0;
}