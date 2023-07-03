import cv2

from src.utils.logger import Log
from src.utils.util import num2pos
from src.utils.detect import detect_person, detect_ball


class VideoLoader():
    def __init__(self, url, gap):
        self.videoCapture = cv2.VideoCapture(url)
        Log.info("视频帧率: " + str(self.videoCapture.get(cv2.CAP_PROP_FPS)))
        self.cnt = 0
        while True:
            self.cnt += 1
            while self.cnt % gap != 0:
                success, frame = self.videoCapture.read()
                if not success:
                    break
                self.cnt += 1
            # cv2.imshow("ii", frame)
            # cv2.waitKey(0)
            candidate, person = detect_person(frame)
            ball = detect_ball(frame)
            try:
                if self._satisfy(candidate, person, ball):
                    break
            except Exception as e:
                # traceback.print_exc()
                Log.debug("未识别到所需信息")

        self.gap = gap
        self.round = 1
        self.sustaining = False
        Log.debug("视频已定位到垫球动作开始位置，从第%d帧开始检测" % self.cnt)

    def _satisfy(self, candidate, person, ball):
        return False

    def __iter__(self):
        # self.cnt = 0
        return self

    def __next__(self):
        self.cnt += 1
        while self.cnt % self.gap != 0:
            success, frame = self.videoCapture.read()
            if not success:
                raise StopIteration
            self.cnt += 1

        candidate, person = detect_person(frame)
        ball = detect_ball(frame)

        # 该函数处于动作识别过程中，若未检测到关键点应该直接将该异常抛给更高层
        try:
            result = self._satisfy(candidate, person, ball)
        except:
            Log.error("第%d帧存在关键点无法检测的行为" % self.cnt)
            return self.__next__()

        if result and self.sustaining:
            self.round += 1
            self.sustaining = False

        self.sustaining = result

        return candidate, person, ball, frame, self.round


class DigVideoLoader(VideoLoader):
    def __init__(self, url, gap):
        super().__init__(url, gap)

    # 1. 球位于小臂之间
    # 2. 球与手臂保持水平
    def _satisfy(self, candidate, person, ball):
        left = self.__judge([3, 4], candidate, person, ball)
        right = self.__judge([6, 7], candidate, person, ball)

        return left or right

    def __judge(self, pos, candidate, person, ball):
        pos_x = pos[0]
        pos_y = pos[1]
        p1 = num2pos(pos_x, candidate, person)
        p2 = num2pos(pos_y, candidate, person)

        if ball is None: return False
        ball_circle = (ball[0] + ball[2]) / 2
        x = [p1[0], p2[0], ball_circle]
        y = [p1[1], p2[1], ball[3]]
        x.sort(), y.sort()
        Log.debug("x" + x.__str__() + " " + "y" + y.__str__())

        return x[1] == ball_circle and y[2] - y[0] <= 20
