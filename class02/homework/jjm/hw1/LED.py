
# Arduino peripheral controll

from iotdemo import FactoryController
import time


with FactoryController('/dev/ttyACM0') as ctrl:

    while(1):
        ctrl.red = False
        ctrl.orange = False
        ctrl.green = False

        ctrl.push_actuator(1) 
        ctrl.push_actuator(2) 

        ctrl.conveyor = True

        time.sleep(3)

        ctrl.close()
        time.sleep(1)



