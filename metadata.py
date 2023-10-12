# Written by Uglješa Lukešević (github.com/ukicomputers)
# TODO: normalizeImage

MODEL_TYPE = "yolo_nas_l"  # change this variable to your model type

MODEL_TRAINED_CLASSES = (
    2  # change this variable to num. of classes that your model is trained
)

MODEL_PATH = "./model.pth"  # change this variable to your model path

import numpy as np

std = None
brp = None
cp = None
dr = None
dlmr = None


def get_preprocessing_steps(preprocessing, processing):
    global std, brp, cp, dr, dlmr
    if isinstance(preprocessing, processing.StandardizeImage):
        std = preprocessing.max_value
    elif isinstance(preprocessing, processing.DetectionRescale):
        dr = True
    elif isinstance(preprocessing, processing.DetectionLongestMaxSizeRescale):
        dlmr = True
    elif isinstance(preprocessing, processing.DetectionBottomRightPadding):
        brp = preprocessing.pad_value
    elif isinstance(preprocessing, processing.DetectionCenterPadding):
        cp = preprocessing.pad_value


def main():
    global std, brp, cp, dr, dlmr
    from super_gradients.training import models
    import super_gradients.training.processing as processing

    net = models.get(
        MODEL_TYPE, num_classes=MODEL_TRAINED_CLASSES, checkpoint_path=MODEL_PATH
    )

    dummy = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

    iou = net._default_nms_iou
    conf = net._default_nms_conf

    for st in net._image_processor.processings:
        get_preprocessing_steps(st, processing)

    imgsz = np.expand_dims(net._image_processor.preprocess_image(dummy)[0], 0).shape

    file = (
        str(iou) + "\n" + str(conf) + "\n" + str(imgsz[2]) + "\n" + str(imgsz[3]) + "\n"
    )

    if std != None:
        file += str(std) + "\n"
    else:
        file += "n" + "\n"

    if dr != None:
        file += "t" + "\n"
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
        file += "n"

    """
    metadata output:
        iou thereshold (float)
        score thereshold (float)
        width (int)
        height (int)
        standardize (string t:n, 1:0)
        detect rescale (string t:n, 1:0)
        detect long max rescale (string t:n, 1:0)
        bottom right padding (int:string (if n))
        center padding (int:string (if n))
    """

    filename = f"metadata"
    with open(filename, "w") as f:
        f.write(file)

    models.convert_to_onnx(model=net, input_shape=imgsz[1:], out_path="model.onnx")


if __name__ == "__main__":
    main()
