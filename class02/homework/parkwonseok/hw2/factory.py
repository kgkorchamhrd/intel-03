#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
# from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector #모션디텍터 추가

FORCE_STOP = False


def thread_cam1(q):
    
    # TODO: MotionDetector
    cNotion = MotionDetector()
    cNotion.load_preset('motion.cfg')
    # TODO: Load and initialize OpenVINO

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put( ("VIDEO:Cam1 live", frame))

        # TODO: Motion detect
        detected_frame = cNotion.detect(frame)
        if detected_frame is not None:
            q.put(("VIDEO:Cam1 detected", detected_frame))  # 감지된 영상을 큐에 넣어야하나

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    cNotion = MotionDetector()
    cNotion.load_preset('motion.cfg')

    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        #추가한 부분
        q.put( ("VIDEO:Cam2 live", frame) )
        
        
        # TODO: Detect motion
        #추가한 부분 ( 모션 디텍션 )
        detected_frame = cNotion.detect(frame)
        if detected_frame is not None:
            q.put(("VIDEO:Cam2 detected", detected_frame)) 

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.


        # TODO: Detect color

        # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # TODO: HW2 Create a Queue
    #추가한 부분분
    q = Queue()  # 🔹 멀티쓰레딩을 위한 큐 생성

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    #추가한 부분
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()


    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            #추가한 부분
            try:
                name, data = q.get(timeout=1)  # 🔹 큐에서 데이터 가져오기
            except Empty:
                continue  # 큐가 비어 있으면 다시 루프 실행

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            #추가한 부분
            if name == "VIDEO:Cam1 live":
                imshow("Cam1 live", data, pos=(100, 100))
            elif name == "VIDEO:Cam2 live":
                imshow("Cam2 live", data, pos=(500, 100))
            elif name == "VIDEO:Cam1 detected":
                imshow("Cam1 detected", data, pos=(100, 500))
            elif name == "VIDEO:Cam2 detected":
                imshow("Cam2 detected", data, pos=(500, 400))

            # TODO: Control actuator, name == 'PUSH'

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()
    #조인 추가
    t1.join()
    t2.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(1) #코드 반환?
