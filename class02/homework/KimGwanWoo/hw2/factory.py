#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
#from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('resources/motion.cfg')

    # TODO: Load and initialize OpenVINO


    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # TODO: Motion detect
        detected = cMotion.detect(frame)


        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        if detected is not None:
            q.put(("VIDEO:Cam1 detected", detected))

            # abnormal detect
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reshaped = detected[:, :, [2, 1, 0]]
            np_data = np.moveaxis(reshaped, -1, 0)
            preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
            batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        #print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('motion.cfg')


    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info

        # TODO: Detect motion

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 live", frame))

        # TODO: Detect color





        # TODO: Compute ratio
        #print(f"{name}: {ratio:.2f}%")

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
    q = Queue()

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    cam1_thread = threading.Thread(target=thread_cam1, args=(q,))
    cam2_thread = threading.Thread(target=thread_cam2, args=(q,))
    cam1_thread.start()
    cam2_thread.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            name, frame = q.get(timeout=1)

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.


            if "live" in name:
                imshow(name[6:], frame)
            elif "detected" in name:
                imshow(name[6:], frame)

            # if name == "VIDEO:Cam1 live":
            #     imshow("Cam1 live", frame, pos=(0, 0))
            # elif name == "VIDEO:Cam2 live":
            #     imshow("Cam2 live", frame, pos=(640, 0))

            # TODO: Control actuator, name == 'PUSH'
            if name == 'PUSH':
                ctrl.push() 


            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()
            
    cam1_thread.join()
    cam2_thread.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
