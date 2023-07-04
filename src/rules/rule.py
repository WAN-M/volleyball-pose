from src.enums.action import Action
from src.rules.dig import digRule


class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, image, candidate, person, ball, result,*args, **kwargs):
        # 按顺序调用具体规则
        if self.type is Action.Dig:
            return digRule.sum_rules(image, candidate, person, ball, result)
        else:
            return "暂不支持该动作"
