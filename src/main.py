import cv2

from src.enum.action import Action
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
    # open the video and get frames
    videoCapture = cv2.VideoCapture(url)
    i = 0
    while True:
        success, frame = videoCapture.read()
        if not success:
            break
        i += 1
        if i % 100 == 0:
            handle_picture(frame)

# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    solve("E:\\cv\\volleyball-pose\\videos\\KUN.mp4")
