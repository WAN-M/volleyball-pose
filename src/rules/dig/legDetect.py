from src.utils.logger import Log
from src.utils.util import num2pos, draw_wrong_place


def get_points(points, candidates, persons):
    y = []
    for i in range(len(candidates)):
        y.append(dict())
        for point in points:
            y[i][point] = num2pos(point, candidates[i], persons[i])
    return y


# 1. 大腿y坐标的变化大于最大值的30%
# 2. 大腿y坐标在一轮里先大后小
def one_leg(images, nums, candidates, persons) -> bool:
    dis = []
    for i in range(len(persons)):
        try:
            d = num2pos(nums[1], candidates[i], persons[i]) - \
                num2pos(nums[0], candidates[i], persons[i])
        except:
            continue
        dis.append(d[1])
    if len(dis) < 3:
        Log.info("腿部有效信息不足")
        return True

    flag = max(dis) - min(dis) > max(dis) * 0.3
    up = True
    for i in range(1, len(dis)):
        if dis[i] - dis[i - 1] > 0:
            if not up:
                flag = False
        else:
            if up:
                up = False

    return flag


# 腿每次应该上下起伏
def detect(images, candidates, persons) -> bool:
    flag = True
    if not one_leg(images, [8, 9], candidates, persons):
        for i in range(len(images)):
            try:
                draw_wrong_place(images[i], *num2pos(8, candidates[i], persons[i]))
            except:
                continue
        flag = False
    if not one_leg(images, [11, 12], candidates, persons):
        for i in range(len(images)):
            try:
                draw_wrong_place(images[i], *num2pos(11, candidates[i], persons[i]))
            except:
                continue
        flag = False
    return flag
