import time
from adafruit_servokit import ServoKit
import threading

arm = ServoKit(channels=16)

arm.servo[0].set_pulse_width_range(600,2600)
arm.servo[2].set_pulse_width_range(600,2600)
arm.servo[4].set_pulse_width_range(600,2600)
arm.servo[6].set_pulse_width_range(600,2600)
arm.servo[8].set_pulse_width_range(600,2600)
arm.servo[10].set_pulse_width_range(600,2600)

arm.servo[0].actuation_range = 180
arm.servo[2].actuation_range = 180
arm.servo[4].actuation_range = 180
arm.servo[6].actuation_range = 180
arm.servo[8].actuation_range = 180
arm.servo[10].actuation_range = 180
    
def move(channel, position):
    current = round(arm.servo[channel].angle)
    if current<position:
        for x in range(current, position, 1):
            arm.servo[channel].angle = x
            time.sleep(0.05)
    else:
        for x in range(current, position, -1):
            arm.servo[channel].angle = x
            time.sleep(0.05)

def smove(pos0, pos2, pos4, pos6): 
    t1= threading.Thread(target=move, args=(0,pos0))
    t2= threading.Thread(target=move, args=(2,pos2))
    t3= threading.Thread(target=move, args=(4,pos4))
    t4= threading.Thread(target=move, args=(6,pos6))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

def e_start():
    smove(45, 10, 180, 70)
    smove(45, 90, 180, 126)
    move(10, 50)
    
def init():
    arm.servo[0].angle = 45
    arm.servo[2].angle = 60
    arm.servo[4].angle = 30
    arm.servo[6].angle = 130
    arm.servo[8].angle = 25
    arm.servo[10].angle = 40

def put():
    move(8,25)
    move(10,40)
    smove(45, 60, 30, 126)
    smove(45, 126, 30, 70)
    smove(6, 100, 56, 70)
    smove(6, 70, 66, 90)
    move(10, 10)
    smove(6, 100, 56, 70)
    smove(45, 126, 30, 70)
    smove(45, 60, 30, 130)
    smove(100, 60, 30, 132)

def motion_1():
    smove(45, 60, 30, 130)
    smove(45, 126, 30, 70)
    smove(6, 100, 56, 70)
    smove(6, 70, 66, 90)
    move(10, 40)
    smove(6, 100, 56, 70)
    smove(45, 126, 30, 70)

def motion_2():
    smove(45, 60, 30, 130)
    smove(45, 120, 20, 90)
    smove(6, 120, 10, 90)
    smove(6, 90, 20, 116)
    smove(6, 50, 50, 116)
    move(10,40)
    smove(6, 90, 20, 116)
    
def idle():
    smove(45, 60, 30, 130)


# e_start()    
    
init()
# put()
# time.sleep(3)
# motion_1()
# idle()
# put()
# time.sleep(3)
# motion_2()
# idle()