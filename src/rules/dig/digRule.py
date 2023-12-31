from src.rules.dig import armDetect, armTorsoAngle, legDetect
from src.rules.dig.ball_position import ball_position
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages

import cv2


def sum_rules(images, candidates, persons, balls):
    # 需要每张图片遍历的规则
    mes = set()
    for i in range(len(images)):
        # 每张图片需要遵守的规则
        if not armTorsoAngle.detect(images[i], candidates[i], persons[i]):
            mes.add("手臂与躯干最大角度不应超过100°")

        # 击球前后需要遵守的规则
        try:
            if arm_dis_ball(candidates[i], persons[i], balls[i]) < 2:
                if not ball_position(images[i], candidates[i], persons[i], balls[i]):
                    mes.add("击球时球离手腕位置太远")
                if not armDetect.detect_arm_status(images[i], candidates[i], persons[i]):
                    mes.add("手臂没有伸直")
        except:
            Log.error("球未被识别")

        if len(mes) > 0:
            image = draw_messages(images[i], mes)
            images[i] = image

    # 需要整体判断的规则
    if len(images) > 1:     # 不是图片
        mes2 = set()
        if not legDetect.detect(images, candidates, persons):
            mes2.add("腿部动作有误")

        if len(mes2) > 0:
            mes.add("腿部动作有误")
            for i in range(len(images)):
                image = images[i]
                image2 = draw_messages(image, mes2)
                images[i] = image2

    return mes
