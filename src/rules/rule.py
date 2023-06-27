from src.enum.action import Action
from src.rules import digRule


class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, image, candidate, subset, *args, **kwargs):
        # 按顺序调用具体规则
        if self.type is Action.Dig:
            digRule.sum_rules(image, candidate, subset)
        else:
            print("暂不支持该动作")
