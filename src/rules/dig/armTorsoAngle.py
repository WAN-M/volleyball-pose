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

    print("angle of arm and torso is: %f" % angle)
    return angle <= 100


def detect(image, candidate, person) -> bool:
    flag = True
    # print(candidate[person[5]][0:2])
    # 判断左臂与躯干，选取5,6,11
    if not judge_angle(*util.num2pos(5, candidate, person),
                       *util.num2pos(6, candidate, person),
                       *util.num2pos(11, candidate, person)):
        flag = False
        util.draw_wrong_place(image, *util.num2pos(5, candidate, person))
    # 判断右臂与躯干，选取2,3,8
    if not judge_angle(*util.num2pos(2, candidate, person),
                       *util.num2pos(3, candidate, person),
                       *util.num2pos(8, candidate, person)):
        flag = False
        util.draw_wrong_place(image, *util.num2pos(2, candidate, person))
    return flag
