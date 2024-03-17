# Written by Uglješa Lukešević (github.com/ukicomputers)
# Script for writing required metadata used in yolonas-cpp library

MODEL_TYPE = "yolo_nas_s"  # change this variable to your model type

MODEL_TRAINED_CLASSES = (
    2  # change this variable to num. of classes that your model is trained
)

MODEL_PATH = "./model.pth"  # change this variable to your model path
CONVERT_TO_ONNX = True  # do you want to convert that model to ONNX

#################################################################################
from super_gradients.training import models
import super_gradients.training.processing as processing
import numpy as np

std = None
brp = None
cp = None
dlmr = None
norm = None

def get_preprocessing_steps(preprocessing, processing):
    global std, brp, cp, dlmr, norm
    if isinstance(preprocessing, processing.StandardizeImage):
        std = preprocessing.max_value
    elif isinstance(preprocessing, processing.DetectionLongestMaxSizeRescale):
        dlmr = True
    elif isinstance(preprocessing, processing.DetectionBottomRightPadding):
        brp = preprocessing.pad_value
    elif isinstance(preprocessing, processing.DetectionCenterPadding):
        cp = preprocessing.pad_value
    elif isinstance(preprocessing, processing.NormalizeImage):
        norm = {preprocessing.mean.toList(), std.mean.toList()}

def main():
    global std, brp, cp, dlmr, norm

    net = models.get(
        MODEL_TYPE, num_classes=MODEL_TRAINED_CLASSES, checkpoint_path=MODEL_PATH
    )

    dummy = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

    for st in net._image_processor.processings:
        get_preprocessing_steps(st, processing)

    imgsz = np.expand_dims(net._image_processor.preprocess_image(dummy)[0], 0).shape

    file = (
        str(net._default_nms_iou) + "\n" + 
        str(net._default_nms_conf) + "\n" + 
        str(imgsz[2]) + "\n" + 
        str(imgsz[3]) + "\n"
    )

    if std != None:
        file += str(std) + "\n"
    else:
        file += "n" + "\n"

    if dlmr != None:
        file += "t" + "\n"
    else:
        file += "n" + "\n"

    if brp != None:
        file += str(brp) + "\n"
    else:
        file += "n" + "\n"

    if cp != None:
        file += str(cp) + "\n"
    else:
        file += "n" + "\n"
        
    if norm != None:
        file += "t" + "\n"
        file += str(norm[0][0]) + "\n"
        file += str(norm[0][1]) + "\n"
        file += str(norm[0][2]) + "\n"
        file += str(norm[1][0]) + "\n"
        file += str(norm[1][1]) + "\n"
        file += str(norm[1][2])
    else:
        file += "n"

    """
    metadata output:
        iou thereshold (float)
        score thereshold (float)
        width (int)
        height (int)
        standardize (string t:n, 1:0)
        detect long max rescale (string t:n, 1:0)
        bottom right padding (int:string (if n))
        center padding (int:string (if n))
        normalize (string t:n, 1:0)
        std[0] (int)
        std[1] (int)
        std[2] (int)
        mean[0] (int)
        mean[1] (int)
        mean[2] (int)
    """

    filename = f"metadata"
    with open(filename, "w") as f:
        f.write(file)

    if CONVERT_TO_ONNX == True:
        models.convert_to_onnx(model=net, input_shape=imgsz[1:], out_path="model.onnx")

if __name__ == "__main__":
    main()
