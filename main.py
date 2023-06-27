import cv2
from matplotlib import pyplot as plt
from bottle import Bottle, request

from src.enums.action import Action
from src.models.body import Body
from src.result.result import CommonResult
from src.rules.rule import Rule

debug = False

body_estimation = Body('./model/body_pose_model.pth')
# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

app = Bottle()


@app.route('/cv', method='POST')
def process():
    url = request.json
    try:
        messages, images = solve(url)
    except Exception as e:
        print(e)
        return CommonResult.fail(e)
    else:
        print("完成请求: %s" % url)
        # plt.imshow(images[0][:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()
        # return CommonResult.success(messages, images)

    # img1 = cv2.imread("./images/vol.png")
    # img2 = cv2.imread("./images/hand_preview.png")
    # return CommonResult.success("success", img1)
    # return CommonResult.success("message", "data")


# 帧中可能有多个人，从中选出需要分析的人
def select_person(subset):
    sort_subset = sorted(subset, key=lambda x: (-x[-2] / x[-1], -x[-1]))
    # print(sort_subset)
    return sort_subset[0]


# 返回图片中不标准的姿势信息，并在图片上标出位置
def handle_picture(image):
    candidate, subset = body_estimation(image)
    # 利用规则判断，并在图片上绘出不标准点
    person = select_person(subset)
    return rule(image, candidate, person)


def solve(url):
    # 存储所有姿态不标准信息
    all_mes = set()
    all_img = []

    # 打开视频并抽取需要的帧识别
    videoCapture = cv2.VideoCapture(url)
    i = 0
    while True:
        success, frame = videoCapture.read()
        if not success:
            break
        i += 1
        if i % 500 == 0:
            try:
                pic_mes = handle_picture(frame)
            except Exception as e:
                print(e)
                continue

            # 若当前帧中人物姿态出现了之前未出现的信息，则返回该图片
            flag = False
            for message in pic_mes:
                if message not in all_mes:
                    all_mes.add(message)
                    flag = True
            if flag:
                all_img.append(frame)

            # plt.imshow(frame[:, :, [2, 1, 0]])
            # plt.axis('off')
            # plt.show()
            # break

    # 不存在有效图像
    if len(all_img) == 0:
        raise Exception("视频不符合要求")
    return list(all_mes), all_img


# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    print("项目已启动")
    if debug:
        solve("../videos/KUN.mp4")
    else:
        app.run(host='localhost', port=5000)
