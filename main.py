import copy
import logging
from pathlib import Path

import cv2
from bottle import Bottle, request
from matplotlib import pyplot as plt

from src.enums.action import Action
from src.models.ball import detect_ball
from src.models.body import Body
from src.models.common import DetectMultiBackend
from src.result.result import CommonResult
from src.rules.rule import Rule
from src.utils.dataloaders import VID_FORMATS, IMG_FORMATS
from src.utils.logger import Log
from src.utils.torch_utils import select_device
from src.utils.util import draw_wrong_place

debug = True

body_estimation = Body('./model/body_pose_model.pth')

device = select_device()
volleyball_model = DetectMultiBackend('./model/yolov5x.pt', device=device, dnn=False, data='data/coco128.yaml')
# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

app = Bottle()
logging.getLogger('log').setLevel(logging.WARNING)


@app.route('/cv', method='POST')
def process():
    url = request.json
    client_ip = request.environ.get('REMOTE_ADDR')
    Log.info("IP: " + client_ip + " FILE: " + url)
    try:
        messages, images = solve(url)
    except Exception as e:
        Log.error(e)
        return CommonResult.fail(e)
    else:
        Log.info("完成请求: %s" % url)
        # plt.imshow(images[0][:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()
        return CommonResult.success(messages, images)

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
    image0 = copy.deepcopy(image)
    candidate, subset = body_estimation(image0)
    ball_pose = detect_ball(volleyball_model, image0)
    draw_wrong_place(image, int(ball_pose[0]), int(ball_pose[1]))
    draw_wrong_place(image, int(ball_pose[2]), int(ball_pose[3]))
    plt.imshow(image[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()
    # Log.info("(%d %d), (%d, %d)" %(int(ball_pose[0]), int(ball_pose[1]), int(ball_pose[2]), int(ball_pose[3])))
    # 利用规则判断，并在图片上绘出不标准点
    person = select_person(subset)
    return rule(image, candidate, person)


def solve(url):
    # 存储所有姿态不标准信息
    all_mes = set()
    all_img = []
    if Path(url).suffix[1:] in VID_FORMATS:
        # 打开视频并抽取需要的帧识别
        videoCapture = cv2.VideoCapture(url)
        i = 0
        while True:
            success, frame = videoCapture.read()
            if not success:
                break
            i += 1
            if i % 100 == 0:
                # try:
                #     pic_mes = handle_picture(frame)
                # except Exception as e:
                #     Log.error(e)
                #     continue
                pic_mes = handle_picture(frame)

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
    # 上传的图片
    elif Path(url).suffix[1:] in IMG_FORMATS:
        image = cv2.imread(url, 1)
        pic_mes = ""
        try:
            pic_mes = handle_picture(image)
        except Exception as e:
            print(e)
        if len(pic_mes) > 0:
            all_mes = pic_mes
            all_img.append(image)
    # 不存在有效图像
    else:
        raise Exception("上传的文件不符合要求")
    return list(all_mes), all_img


# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    Log.info("项目已启动")
    if debug:
        solve("./images/vol.png")
    else:
        app.run(host='localhost', port=5000, )
