from src.rules.dig import armDetect, armTorsoAngle


def sum_rules(image, candidate, person):
    mes = set()
    if not armDetect.detect_arm_status(image, candidate, person):
        mes.add("手臂未保持直线")
    if not armTorsoAngle.detect(image, candidate, person):
        mes.add("手臂与躯干最大角度不应超过100°")
    return mes
