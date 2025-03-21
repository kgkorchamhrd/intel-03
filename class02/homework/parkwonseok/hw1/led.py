from iotdemo import FactoryController

ctrl = FactoryController('dev/ttyACM0')

from iotdemo import FactoryController
from time import sleep


with FactoryController('/dev/ttyACM0') as ctrl:

    while(1):

        ctrl.red = False
        ctrl.orange = False
        ctrl.green = False

        # ctrl.push_actuator(1) 
        # ctrl.push_actuator(2)
        # ctrl.push_actuator(3)  
        # ctrl.conveyor = True
        ctrl.close()

