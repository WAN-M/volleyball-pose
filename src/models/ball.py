import numpy as np
import torch

from src.utils.general import Profile, non_max_suppression, scale_boxes
from src.utils.augmentations import letterbox

ROOT = ""


def run(
        model,
        image,  # file/dir/URL/glob/screen/0(webcam)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
):
    # Load model
    stride = model.stride
    imgsz = (640, 640)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    # im为resize + pad之后的图片
    im = letterbox(image, imgsz, stride=stride)[0]  # resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    with Profile():
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with Profile():
        pred = model(im)

    # NMS
    with Profile():
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    temp_xyxy = None
    temp_xyxy2 = None
    for i, det in enumerate(pred):  # per image

        im0 = image.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            max_size = 0
            max_size2 = 0
            for *xyxy, conf, cls in reversed(det):
                length = abs(xyxy[2] - xyxy[0])
                if (int(cls) == 32 and length > max_size):
                    temp_xyxy = xyxy
                    max_size = length
                if (int(cls) == 38 and length > max_size2):
                    width = abs(xyxy[3] - xyxy[1])
                    if length / width > 0.8 and length / width < 1.25:
                        temp_xyxy2 = xyxy
                        max_size2 = length
    if temp_xyxy is not None:
        return [x.item() for x in temp_xyxy]
    elif temp_xyxy2 is not None:
        return [x.item() for x in temp_xyxy2]
    else:
        return None
