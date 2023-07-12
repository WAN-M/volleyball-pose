import os
from pathlib import Path
from queue import Queue

import cv2

from src.utils.detect import detect_person, detect_ball, ROOT
from src.utils.logger import Log
from src.utils.util import arm_dis_ball



class VideoLoader():
    output_num = "001"
    origin_format = output_num + ".avi"
    output_format = output_num + ".mp4"
    def __init__(self, url):
        self.gap = 2
        self.videoCapture = cv2.VideoCapture(url)
        self.cnt = 0
        self.fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))
        self.output_path = str(ROOT) + "/output/" + Path(url).stem
        self.video = cv2.VideoWriter(self.output_path + VideoLoader.output_format,
                                     cv2.VideoWriter_fourcc(*'MP4V'),
                                     self.fps,
                                     (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.frames = dict()        # 视频的每一帧
        self.detect_num = []        # 选取了哪些帧号的帧用于检测
        self.p = 0                  # detect_num的指针
        Log.info("视频帧率:%d " % self.fps)
        loop_cnt = 0
        while True:
            while loop_cnt < self.gap:
                success, frame = self.videoCapture.read()
                if not success:
                    self.close()
                    break
                # 开始的帧都视为未检测的
                self.add_frame(frame)
                loop_cnt += 1
                self.cnt += 1
            Log.debug("第%d帧检测开始" % self.cnt)
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

    def add_frame(self, frame, detect=False):
        if detect:
            num = self.detect_num[self.p - 1]
            while num < self.detect_num[self.p]:
                self.video.write(self.frames[num])
                num += 1
            self.p += 1

            for i in range(self.fps):
                self.video.write(frame)
        else:
            # self.video.write(frame)
            # 将帧先存起来，后续完成需要检测的帧后再将前面的帧一同写入
            self.frames[self.cnt] = frame

    def close(self):
        self.video.release()
        Log.info("结果视频已生成")

        # 将cv2导出的avi格式转成mp4格式
        # os.system(f'ffmpeg -i "{self.output_path + self.origin_format}" -vcodec h264 "{self.output_path + self.output_format}"')

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

    def get_all_pic(self):
        candidates = []
        persons = []
        balls = []
        frames = []
        for candidate, person, ball, frame, round, result in self:
            candidates.append(candidate)
            persons.append(person)
            balls.append(ball)
            frames.append(frame)
            self.add_frame(frame, True)
            if round > 1: break
        # 用于写入视频初始化
        self.detect_num.append(0)
        return candidates, persons, balls, frames

    def __iter__(self):
        # self.cnt = 0
        return self

    def __next__(self):
        loop_cnt = 0
        while True:
            success, frame = self.videoCapture.read()
            if not success:
                self.close()
                raise StopIteration
            loop_cnt += 1
            self.cnt += 1
            if loop_cnt < self.gap:
                self.add_frame(frame)
            else:
                break

        self.detect_num.append(self.cnt)
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