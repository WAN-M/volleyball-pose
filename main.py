from pathlib import Path

import cv2
from bottle import Bottle, request

from src.enums.action import Action
from src.result.result import CommonResult
from src.rules.rule import Rule
from src.utils.dataloaders import DigVideoLoader
from src.utils.detect import detect_person, detect_ball
from src.utils.logger import Log

debug = True

# 目前只做垫球，后续可拓展
rule = Rule(Action.Dig)

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

app = Bottle()


@app.route('/cv', method='POST')
def process():
    url = request.json
    client_ip = request.environ.get('REMOTE_ADDR')
    Log.info("IP: " + client_ip + " FILE: " + url)
    try:
        result = solve(url)
        Log.info("完成请求: %s" % url)
        return result
    except Exception as e:
        Log.error(str(e))
        return CommonResult.fail(str(e))


def solve(url):
    # 存储所有姿态不标准信息
    all_mes = set()
    all_img = []
    if Path(url).suffix[1:] in VID_FORMATS:
        videoLoader = DigVideoLoader(url)
        candidates, persons, balls, frames = videoLoader.get_all_pic()
        rule(frames, candidates, persons, balls)
        # videoLoader.test()
        print("图像共%d张" % len(frames))
        for frame in frames:
            videoLoader.add_frame(frame, True)
        videoLoader.close()
        # 只返回生成的视频
        return CommonResult.success(None, None, videoLoader.output_path)
    # 上传的图片
    elif Path(url).suffix[1:] in IMG_FORMATS:
        image = cv2.imread(url, 1)
        candidate, person = detect_person(image)
        ball = detect_ball(image)
        pic_mes = rule([image], [candidate], [person], [ball])
        if len(pic_mes) > 0:
            all_mes = pic_mes
            all_img.append(image)

        return CommonResult.success(all_mes, all_img)
    # 不存在有效图像
    else:
        raise Exception("上传的文件不符合要求")


# 项目总入口，传入视频进行处理
if __name__ == '__main__':
    Log.info("项目已启动")
    if debug:
        solve("./videos/standard.mp4")
    else:
        app.run(host='localhost', port=5000)
