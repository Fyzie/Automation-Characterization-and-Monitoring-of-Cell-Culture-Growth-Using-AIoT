import adafruit_dht
import adafruit_vcnl4040
import board
import time

while True:
    try:
        dht_pin = board.D4
        i2c = board.I2C()

        dht = adafruit_dht.DHT22(dht_pin, use_pulseio=False)
        vcnl = adafruit_vcnl4040.VCNL4040(i2c)
        temperature = dht.temperature
        humidity = dht.humidity
        proximity = vcnl.proximity
        light = vcnl.light
        white = vcnl.white
#         print('{:.2f}'.format(proximity))
        if proximity >= 10:
            door = "Closed"
        else:
            door = "Opened"
        door_position = door
        if humidity is not None and temperature is not None:
            print('T: {:.2f}*C |'.format(temperature), end=' ')
            print('H: {:.2f}% |'.format(humidity), end=' ')
            print('W: {:.2f} |'.format(white), end=' ')
            print('L: {:.2f} |'.format(light), end=' ')
            print('Door: ', door)
            
    except RuntimeError as e:
        pass
    time.sleep(1)
