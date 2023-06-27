

class CommonResult():
    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data

    def success(self, data):
        return CommonResult(200, "操作成功", data)

    def fail(self, message):
        return CommonResult(500, message, None)