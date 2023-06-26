
class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, image, candidate, subset, *args, **kwargs):
        # 按顺序调用具体规则
        print("enter")
