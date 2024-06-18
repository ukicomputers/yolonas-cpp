// Written by Uglješa Lukešević (github.com/ukicomputers)
// This application IS NOT MEANT TO BE USED WITH NORMAL PC! It is used for Docker container only.

#include <ukicomputers/YoloNAS.hpp>
#include <cstdlib>
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

int main()
{
    YoloNAS net(getenv("model"), getenv("metadata"), COCO_LABELS);
    cv::Mat img = cv::imread(getenv("source"));
    auto result = net.predict(img);

    for (int i = 0; i < result.size(); i++)
    {
        cout << "************" << endl;
        cout << "Detected: " + result[i].label << endl;
        cout << "Score: " + to_string(result[i].score) << endl;
        cout << "X, Y: " + to_string(result[i].x) + ", " + to_string(result[i].y) << endl;
        cout << "W, H: " + to_string(result[i].w) + ", " + to_string(result[i].h) << endl;
    }

    cv::imwrite("/output/detected.jpg", img);
    return 0;
}
