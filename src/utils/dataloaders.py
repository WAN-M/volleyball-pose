import cv2
import math
import numpy as np

from src.utils.logger import Log
from src.utils.util import arm_dis_ball
from src.utils.detect import detect_person, detect_ball


class VideoLoader():
    def __init__(self, url):
        self.gap = 2
        self.videoCapture = cv2.VideoCapture(url)
        Log.info("视频帧率: " + str(self.videoCapture.get(cv2.CAP_PROP_FPS)))
        self.cnt = 0
        loop_cnt = 0
        while True:
            while loop_cnt < self.gap:
                success, frame = self.videoCapture.read()
                if not success:
                    break
                loop_cnt += 1
                self.cnt += 1
            Log.info("第%d帧检测开始" % self.cnt)
            candidate, person = detect_person(frame)
            ball = detect_ball(frame)
            self.set_gap(candidate, person, ball)
            loop_cnt = 0
            try:
                if self._satisfy(candidate, person, ball):
                    break
            except Exception as e:
                # traceback.print_exc()
                Log.debug("未识别到所需信息")
            #cv2.imshow("ii", frame)
            #cv2.waitKey(0)
        self.round = 1
        self.sustaining = False
        Log.debug("视频已定位到垫球动作开始位置，从第%d帧开始检测" % self.cnt)

    def _satisfy(self, candidate, person, ball):
        return False

    def set_gap(self, candidate, person, ball):
        try:
            dis = arm_dis_ball(candidate, person, ball)
            Log.info("球离手的距离是球的半径的%f倍" % dis)
            if dis < 5:
                self.gap = 1
            elif dis < 9:
                self.gap = 3
            else:
                self.gap = 5
        except:
            Log.error("第%d帧存在关键点无法检测的行为" % self.cnt)

    def __iter__(self):
        # self.cnt = 0
        return self

    def __next__(self):
        loop_cnt = 0
        while loop_cnt < self.gap:
            success, frame = self.videoCapture.read()
            if not success:
                raise StopIteration
            loop_cnt += 1
            self.cnt += 1
        Log.info("第%d帧检测开始" % self.cnt)
        candidate, person = detect_person(frame)
        ball = detect_ball(frame)
        self.set_gap(candidate, person, ball)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        # 该函数处于动作识别过程中，若未检测到关键点应该直接将该异常抛给更高层
        try:
            result = self._satisfy(candidate, person, ball)
        except:
            Log.error("第%d帧存在关键点无法检测的行为" % self.cnt)
            return self.__next__()

        if result and self.sustaining:
            self.round += 1
            self.sustaining = False

        self.sustaining = not result

        return candidate, person, ball, frame, self.round, result


class DigVideoLoader(VideoLoader):
    def __init__(self, url):
        super().__init__(url)

    # 1. 球位于小臂之间
    # 2. 球与手臂保持水平
    def _satisfy(self, candidate, person, ball):
        try:
            dis = arm_dis_ball(candidate, person, ball)
            return dis < 2
        except:
            Log.error("第%d帧存在关键点无法检测的行为" % self.cnt)