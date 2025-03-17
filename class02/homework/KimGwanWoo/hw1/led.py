import time
from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')




def traffic_light():
    while True:
        ctrl.red = True
        ctrl.green = True
        ctrl.orange = True

        ctrl.push_actuator(1)
        ctrl.push_actuator(2)
        ctrl.conveyor =True
        time.sleep(1)  # ON
        
        ctrl.red = False
        ctrl.green = False
        ctrl.orange = False
        ctrl.conveyor = False
        time.sleep(1)  # OFF



traffic_light()
