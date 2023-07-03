from pathlib import Path

import cv2
from bottle import Bottle, request

from src.enums.action import Action
from src.result.result import CommonResult
from src.rules.rule import Rule
from src.utils.dataloaders import DigVideoLoader
from src.utils.detect import detect_person, detect_ball
from src.utils.logger import Log

debug = False

# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

# 视频取帧的间隔数
video_gap = 5

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

app = Bottle()


@app.route('/cv', method='POST')
def process():
    url = request.json
    client_ip = request.environ.get('REMOTE_ADDR')
    Log.info("IP: " + client_ip + " FILE: " + url)
    try:
        messages, images = solve(url)
    except Exception as e:
        Log.error(str(e))
        return CommonResult.fail(str(e))
    else:
        Log.info("完成请求: %s" % url)
        return CommonResult.success(messages, images)


def solve(url):
    # 存储所有姿态不标准信息
    all_mes = set()
    all_img = []
    if Path(url).suffix[1:] in VID_FORMATS:
        videoLoader = DigVideoLoader(url, video_gap)

        for candidate, person, ball, frame, round in videoLoader:
            if round > 1: break
            # cv2.imshow("ii", frame)
            # cv2.waitKey(0)
            try:
                pic_mes = rule(frame, candidate, person, ball)
            except Exception as e:
                Log.error(str(e))
                continue

            # 若当前帧中人物姿态出现了之前未出现的信息，则返回该图片
            flag = False
            for message in pic_mes:
                if message not in all_mes:
                    all_mes.add(message)
                    flag = True
            if flag:
                all_img.append(frame)
                # cv2.imshow("ii", frame)
                # cv2.waitKey(0)

    # 上传的图片
    elif Path(url).suffix[1:] in IMG_FORMATS:
        image = cv2.imread(url, 1)
        pic_mes = ""
        try:
            candidate, person = detect_person(image)
            ball = detect_ball(image)
            pic_mes = rule(image, candidate, person, ball)
        except Exception as e:
            Log.error(str(e))
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
        solve("./videos/standard.mp4")
    else:
        app.run(host='localhost', port=5000)
