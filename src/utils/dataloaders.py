import cv2

from src.utils.logger import Log
from src.utils.util import num2pos


class VideoLoader():
    def __init__(self, url):
        from src.utils.detect import detect_person, detect_ball
        self.videoCapture = cv2.VideoCapture(url)
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                break
            # cv2.imshow("ii", frame)
            # cv2.waitKey(0)
            candidate, person = detect_person(frame)
            ball = detect_ball(frame)
            try:
                if self._satisfy_(candidate, person, ball):
                    break
            except Exception as e:
                Log.debug(e)
                Log.debug("未识别到所需信息")
        Log.debug("视频已定位到垫球动作开始位置")

    def _satisfy_(self, candidate, person, ball):
        return False

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        success, frame = self.videoCapture.read()
        if not success:
            raise StopIteration
        self.cnt += 1
        return self.cnt, frame


class DigVideoLoader(VideoLoader):
    def __init__(self, url):
        super().__init__(url)

    # 1. 球位于小臂之间
    # 2. 球与手臂保持水平
    def _satisfy_(self, candidate, person, ball):
        # 垫球时左右臂从侧面看基本重合，先只判断左臂与球的位置关系
        p3 = num2pos(3, candidate, person)
        p4 = num2pos(4, candidate, person)

        ball_circle = (ball[0] + ball[2]) / 2
        x = [p3[0], p4[0], ball_circle]
        y = [p3[1], p4[1], ball[3]]
        Log.debug("x" + x.__str__())
        Log.debug("y" + y.__str__())
        x.sort()
        y.sort()

        return x[1] == ball_circle or y[2] - y[0] <= 10
