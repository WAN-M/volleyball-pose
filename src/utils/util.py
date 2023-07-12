import math

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def point_dis_line(point, line_point1, line_point2):
    # 计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def arm_dis_ball(candidate, person, ball):
    try:
        p1 = num2pos(3, candidate, person)
        p2 = num2pos(4, candidate, person)
    except:
        p1 = num2pos(6, candidate, person)
        p2 = num2pos(7, candidate, person)

    if ball is None:
        raise Exception("未检测到球")
    ball_circle = [(ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2]
    ball_radius = abs(ball[3] - ball[1]) / 2
    ball_to_arm = point_dis_line(ball_circle, p1, p2)
    # Log.info("球离手的距离是%f,球的半径是%f" %(ball_to_arm, ball_radius))
    return ball_to_arm / ball_radius


def draw_wrong_place(img, x, y):
    cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 1)


def draw_messages(img, messages):
    text = ""
    for mes in messages:
        text += mes + " "
    text = text[0:-1]

    # 设置字体参数
    font_path = "font/simhei.ttf"
    font_size = 20
    font_color = (255, 255, 255)  # 白色
    font_style = ImageFont.truetype(
        font_path, font_size, encoding="utf-8"
    )

    # 获取字体宽度和高度
    text_width, text_height = font_style.getsize(text)

    # 计算文字位置
    image_height, image_width, _ = img.shape
    text_x = int((image_width - text_width) / 2)
    text_y = image_height - text_height

    # 在图像上添加文字
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    draw.text((text_x, text_y), text, font_color, font=font_style)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def num2pos(num, candidate, person) -> []:
    index = int(person[num])
    if index == -1:
        # 该输出信息可以根据num细化到具体哪个关键点
        raise Exception("输入图像未包含有效关键点" + str(num))

    return candidate[int(person[num])][0:2]


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


# transfer caffe models to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    fig = Figure(figsize=plt.figaspect(canvas))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]))

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            ax.plot(x, y, 'g.')
            if show_number:
                ax.text(x, y, str(i))
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas


# image drawed by opencv is not good.
def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for ie, e in enumerate(edges):
        if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                     thickness=2)

    for i, keyponit in enumerate(peaks):
        x, y = keyponit
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
