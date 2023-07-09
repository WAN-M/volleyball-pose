from src.rules.dig import armDetect, armTorsoAngle
from src.rules.dig.ball_position import ball_position
from src.utils.logger import Log
from src.utils.util import arm_dis_ball, draw_messages


def sum_rules(image, candidate, person, ball, result):
    mes = set()
    if not armDetect.detect_arm_status(image, candidate, person):
        mes.add("手臂未保持直线")
    if not armTorsoAngle.detect(image, candidate, person):
        mes.add("手臂与躯干最大角度不应超过100°")
    if result:
        try:
            dis = arm_dis_ball(candidate, person, ball)
            if dis < 2:
                if not ball_position(image, candidate, person, ball):
                    mes.add("击球时球离手腕位置太远")
        except:
            Log.error("球未被识别")

    if len(mes) > 0:
        draw_messages(image, mes)

    return mes
