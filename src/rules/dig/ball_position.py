from src.utils.util import num2pos
from src.utils.logger import Log
from src.utils.util import draw_wrong_place
import numpy as np

import cv2
from copy import deepcopy

def ball_position(image, candidate, person, ball):
    try:
        p1 = num2pos(3, candidate, person)
        p2 = num2pos(4, candidate, person)
    except:
        p1 = num2pos(6, candidate, person)
        p2 = num2pos(7, candidate, person)
    ball_circle = [(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2]
    vec1 = ball_circle - p1
    vec2 = ball_circle - p2
    vec3 = p1 - p2
    dis1 = np.dot(vec1, vec3)
    dis2 = np.dot(vec2, vec3)
    # image0 = deepcopy(image)
    # draw_wrong_place(image0, p1[0], p1[1])
    # draw_wrong_place(image0, p2[0], p2[1])
    # draw_wrong_place(image0, ball_circle[0], ball_circle[1])
    # cv2.imshow("image", image0)
    # cv2.waitKey(0)
    # Log.debug("The position of p1,p2 and ball is (%f %f) (%f %f) (%f %f)"
    #           % (p1[0], p1[1], p2[0], p2[1], ball_circle[0], ball_circle[1]))
    # Log.debug("dis2 / dis1 = %f" % abs(dis2 / dis1))
    if abs(dis2 / dis1) > 0.5:
        Log.info("击球位置太离手腕太远")
    Log.debug("球的位置检测完成")

    return abs(dis2 / dis1) < 0.5
