'''
检测手臂是否处于伸直状态
'''
import math

from src.utils import util
from src.utils.logger import Log


def detect_line(list):
    point_1 = list[0]
    point_2 = list[1]
    point_3 = list[2]
    a = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    b = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    angle = math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return angle


def get_armPoint(armIndex, candidate, person):
    list1 = [[] for _ in range(3)]
    for n in range(len(armIndex)):
        # index = int(person[armIndex[n]])
        # #print(index)
        # if index == -1:
        #     raise Exception("输入图像未包含手臂的全部状况")
        # x, y = candidate[index][0:2]
        # list1[n].append(x)
        # list1[n].append(y)
        try:
            list1[n] = util.num2pos(armIndex[n], candidate, person)
        except:
            list1 = None
            break

        #print(list1)
    return list1

def detect_arm_status(image, candidate, person):
    status = True
    arms = [2, 3, 4]
    list1 = get_armPoint(arms, candidate, person)
    leftAngle = 0
    rightAngle = 0
    if list1 != None:
        leftAngle = detect_line(list1)
    arms = [5, 6, 7]
    list2 = get_armPoint(arms, candidate, person)
    if list2 != None:
        rightAngle = detect_line(list2)

    Log.debug("左臂角度为%f, 右臂角度为%f" %(leftAngle, rightAngle))
        # print("The left hand angle is %f, The right hand angle is %f" %(leftAngle, rightAngle))
    if leftAngle < 150 and rightAngle < 150:
        Log.info("手臂不够直")
        # print("The left hand isn't straight enough")
        if list1 is not None:
            util.draw_wrong_place(image, list1[1][0], list1[1][1])
        if list2 is not None:
            util.draw_wrong_place(image, list2[1][0], list2[1][1])
        status = False
    if leftAngle != 0 or rightAngle != 0:
        # print("OK")
        Log.debug("armDetect执行完成")
    return status

