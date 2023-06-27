import cv2
from matplotlib import pyplot as plt
from bottle import Bottle, request

from enums.action import Action
from model.body import Body
from rules.rule import Rule

body_estimation = Body('../model/body_pose_model.pth')
# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

app = Bottle()

@app.route('/cv', method='POST')
def process():
    url = request.json
    try:
        solve(url)
    except Exception as e:
        print(e)
    else:
        print("finish!!!!!!!!!!!")

    return "success"

def handle_picture(image):
    candidate, subset = body_estimation(image)
    # 利用规则判断，并在图片上绘出不标准点
    rule(image, candidate, subset)


def solve(url):
    # 打开视频并抽取需要的帧识别
    videoCapture = cv2.VideoCapture(url)
    i = 0
    while True:
        success, frame = videoCapture.read()
        if not success:
            break
        i += 1
        if i % 500 == 0:
            handle_picture(frame)
            plt.imshow(frame[:, :, [2, 1, 0]])
            plt.axis('off')
            plt.show()
            break

# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
    print("项目已启动")