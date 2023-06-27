'''
检测手臂是否处于伸直状态
'''
from src.utils.util import draw_wrong_place
import math

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


def get_armPoint(armIndex, candidate, subset):
    list1 = [[] for _ in range(3)]
    for n in range(len(armIndex)):
        index = int(subset[0][armIndex[n]])
        #print(index)
        if index == -1:
            print("输入图像未包含手臂的全部状况")
            return None
        x, y = candidate[index][0:2]
        list1[n].append(x)
        list1[n].append(y)
        #print(list1)
    return list1

def detect_arm_status(image, candidate, subset):
    status = True
    arms = [2, 3, 4]
    list1 = get_armPoint(arms, candidate, subset)
    leftAngle = 0
    rightAngle = 0
    if list1 != None:
        leftAngle = detect_line(list1)
    arms = [5, 6, 7]
    list2 = get_armPoint(arms, candidate, subset)
    if list2 != None:
        rightAngle = detect_line(list2)
    if leftAngle != 0 and rightAngle != 0:
        print("The left hand angle is %f, The right hand angle is %f" %(leftAngle, rightAngle))
    if leftAngle < 170 and leftAngle != 0:
        print("The left hand isn't straight enough")
        draw_wrong_place(image, list1[1][0], list1[1][1])
        status = False
    if rightAngle < 170 and rightAngle != 0:
        print("The right hand isn't straight enough")
        draw_wrong_place(image, list2[1][0], list2[1][1])
        status = False
    if leftAngle != 0 and rightAngle != 0:
        print("OK")
    return status
