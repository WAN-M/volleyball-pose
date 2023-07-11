from src.rules.dig import armDetect, armTorsoAngle, legDetect
from src.rules.dig.ball_position import ball_position
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages


def sum_rules(images, candidates, persons, balls):
    # 需要每张图片遍历的规则
    mes = set()
    for i in len(images):
        if not armDetect.detect_arm_status(images[i], candidates[i], persons[i]):
            mes.add("手臂未保持直线")
        if not armTorsoAngle.detect(images[i], candidates[i], persons[i]):
            mes.add("手臂与躯干最大角度不应超过100°")
        if arm_dis_ball(candidates[i], persons[i], balls[i]) < 2:
            try:
                dis = arm_dis_ball(candidates[i], persons[i], balls[i])
                if dis < 2:
                    if not ball_position(images[i], candidates[i], persons[i], balls[i]):
                        mes.add("击球时球离手腕位置太远")
            except:
                Log.error("球未被识别")

        if len(mes) > 0:
            draw_messages(images[i], mes)

    # 需要整体判断的规则
    mes = set()
    if len(images) > 1:     # 不是图片
        if not legDetect.detect():
            mes.add("腿部未蹬地发力")

        if len(mes) > 0:
            for image in images:
                draw_messages(image, mes)
    return mes
