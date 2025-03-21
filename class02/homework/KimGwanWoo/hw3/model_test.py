#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import openvino as ov
import cv2
import numpy as np
#from openvino.inference_engine import IECore
from iotdemo import FactoryController, MotionDetector, ColorDetector


FORCE_STOP = False


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 수치 안정성을 위해 최대값 빼기
    return exp_x / exp_x.sum()

def thread_cam1(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('./resources/motion.cfg')

    # TODO: Load and initialize OpenVINO
    core = ov.Core()
    model_path = '/home/kimgwanwoo/workspace/smart-factory/resources/mobilenet_v3_large_otx/20250318_054913/20250318_071310/exported_model.xml'  # 예제 모델 파일
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)

    flag = True

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put( ("VIDEO:Cam1 live", frame) )

        # TODO: Motion detect
        detected = cMotion.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put( ("VIDEO:Cam1 detected", detected) )

        # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)
        input_tensor = np.expand_dims(detected, 0)

        # TODO: Inference OpenVINO
        if flag is True:
            _, h, w, _ = input_tensor.shape
            ppp.input().tensor() \
                    .set_shape(input_tensor.shape) \
                    .set_element_type(ov.Type.u8) \
                    .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            device_name = 'CPU'
            compiled_model = core.compile_model(model, device_name)
            flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        # Normalize the probabilities using softmax
        normalized_probs = softmax(probs)

        # TODO: Calculate ratios
        x_ratio = normalized_probs[0] * 100
        circle_ratio = normalized_probs[1] * 100
        
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        if x_ratio > 90:
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    cMotion.load_preset('./resources/motion.cfg')

    # TODO: ColorDetector
    cdet = ColorDetector()
    cdet.load_preset("./resources/color.cfg")
    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put( ("VIDEO:Cam2 live", frame) )

        # TODO: Detect motion
        detected = cMotion.detect(frame)
        if detected is None:
            continue

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put( ("VIDEO:Cam2 detected", detected) )

        # TODO: Detect color
        colors = cdet.detect(detected)  
        #       
        # TODO: Compute ratio
        #print(f"{name}: {ratio:.2f}%")
        name, ratio = colors[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        if name == 'blue':
            q.put(("PUSH", 2))

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
    Motion = threading.Thread(target=thread_cam1, args=(q, ))
    Color = threading.Thread(target=thread_cam2, args=(q, ))
    
    Motion.start()
    Color.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.

            try:
                data = q.get_nowait()
            except Empty:
                continue

            # de-queue name and data
            name, value = data

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if "live" in name:
                imshow(name[6:], value)
            
            elif "detected" in name:
                # 모션 감지 결과 표시
                imshow(name[6:], value)
            

            # TODO: Control actuator, name == 'PUSH'
            elif name == "PUSH":
                pass            
            
            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()



    cv2.destroyAllWindows()

    Motion.join()
    Color.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
