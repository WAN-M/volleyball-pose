import json_tricks as json


class CommonResult():
    def __init__(self, code, message, data=None, video=None):
        self.code = code
        self.message = message
        self.data = data
        self.video = video

    @classmethod
    def success(self, message, data, video=None):
        return json.dumps(CommonResult(200, message, json.dumps(data), video))

    @classmethod
    def fail(self, message):
        return json.dumps(CommonResult(500, message))