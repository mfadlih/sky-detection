import serial
import time

ser = serial.Serial('/dev/ttyS0', 57600, timeout=1)

def send_attitude(roll=0, pitch=0, heading=0):
    data = [round(roll, 2), round(pitch, 2), round(heading, 2)]

    string = '#'
    for x in data:
        string += str(x)
        string += ','
    string += '&'
    ser.write(string.encode())
    time.sleep(0.5)
