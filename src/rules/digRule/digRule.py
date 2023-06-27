import src.rules.digRule.armDetect as armDetect
import matplotlib.pyplot as plt


def sum_rules(image, candidate, subset):
    armStatus = armDetect.detect_arm_status(image, candidate, subset)
    print(armStatus)
