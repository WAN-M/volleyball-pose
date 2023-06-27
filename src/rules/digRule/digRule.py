from src.rules.digRule import armDetect
from src.rules.digRule import armTorsoAngle


def sum_rules(image, candidate, subset):
    # if not armDetect.detect_arm_status(image, candidate, subset):
    #     print("手臂未保持直线")
    if not armTorsoAngle.detect(image, candidate, subset):
        print("手臂与躯干最大角度不应超过100°")
