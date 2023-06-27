'''
检测手臂与躯干最大角度不超过100°
'''
import math

from src.utils import util


def judge_angle(xm, ym, x1, y1, x2, y2) -> bool:
    a = math.sqrt(math.pow((xm - x1), 2) + math.pow((ym - y1), 2))
    b = math.sqrt(math.pow((xm - x2), 2) + math.pow((ym - y2), 2))
    c = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

    angle = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    return angle <= 100


def get_pos(num, candidate, subset) -> []:
    return candidate[int(subset[0][num])][0:2]


def detect(image, candidate, subset) -> bool:
    flag = True
    # print(candidate[subset[0][5]][0:2])
    # 判断左臂与躯干，选取5,6,11
    if not judge_angle(*get_pos(5, candidate, subset),
                       *get_pos(6, candidate, subset),
                       *get_pos(11, candidate, subset)):
        flag = False
        util.draw_wrong_place(image, *get_pos(5, candidate, subset))
    # 判断右臂与躯干，选取2,3,8
    if not judge_angle(*get_pos(2, candidate, subset),
                       *get_pos(3, candidate, subset),
                       *get_pos(8, candidate, subset)):
        flag = False
        util.draw_wrong_place(image, *get_pos(2, candidate, subset))
    return flag
