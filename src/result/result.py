import json_tricks as json


class CommonResult():
    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data

    @classmethod
    def success(self, message, data):
        return json.dumps(CommonResult(200, message, json.dumps(data)))

    @classmethod
    def fail(self, message):
        return json.dumps(CommonResult(500, message, None))