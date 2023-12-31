import sys
from pathlib import Path

import numpy as np

from src.models.ball import run
from src.models.body import Body
from src.models.common import DetectMultiBackend
from src.models.hand import Hand
from src.utils import util
from src.utils.logger import Log
from src.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
Log.info("项目运行目录: " + str(ROOT))
body_estimation = Body(str(ROOT / 'model/body_pose_model.pth'))
hand_estimation = Hand(ROOT / 'model/hand_pose_model.pth')
device = select_device()
volleyball_model = DetectMultiBackend(str(ROOT / 'model/yolov5x.pt'), device=device, dnn=False, data='data/coco128.yaml')


# 帧中可能有多个人，从中选出需要分析的人
def detect_person(image):
    candidate, subset = body_estimation(image)
    if len(subset) == 0:
        return None, []
    # util.draw_bodypose(image, candidate, subset)
    sort_subset = sorted(subset, key=lambda x: (-x[-2] / x[-1], -x[-1]))
    # print(sort_subset)
    return candidate, sort_subset[0]


def detect_ball(image):
    return run(volleyball_model, image)

def detect_hand(image, candidate, person):
    hands_list = util.handDetect(candidate, [person], image)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(image[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    return all_hand_peaks