from src.enums.action import Action
from src.rules.dig import digRule


class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, images, candidates, persons, balls, *args, **kwargs):
        # 按顺序调用具体规则
        if self.type is Action.Dig:
            return digRule.sum_rules(images, candidates, persons, balls)
        else:
            return "暂不支持该动作"
