import cv2
from matplotlib import pyplot as plt

from src.enums.action import Action
from src.model.body import Body
from src.rules.rule import Rule

body_estimation = Body('../model/body_pose_model.pth')
# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

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
    try:
        solve("../videos/KUN.mp4")
    except Exception as e:
        print(e)
    else:
        print("finish!!!!!!!!!!!")
