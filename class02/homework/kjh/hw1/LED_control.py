from iotdemo import FactoryController
from time import sleep






with FactoryController('/dev/ttyACM0') as ctrl:

    while(1):
        
        ctrl.red = False
        ctrl.orange = False
        ctrl.green = False

        ctrl.push_actuator(1) 
        ctrl.push_actuator(2) 
        ctrl.conveyor = True

        sleep(2)


        # ctrl.red = True
        # ctrl.orange = True
        # ctrl.green = True

        # ctrl.push_actuator(1)
        # ctrl.push_actuator(2) 

        # ctrl.conveyor = False
        ctrl.close()

        sleep(2)



    