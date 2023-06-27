from enums.action import Action
from rules.digRule import digRule


class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, image, candidate, person, *args, **kwargs):
        # 按顺序调用具体规则
        if self.type is Action.Dig:
            digRule.sum_rules(image, candidate, person)
        else:
            print("暂不支持该动作")
