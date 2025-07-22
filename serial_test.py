import serial
import time

ser = serial.Serial("/dev/ttyS0", 115200)

ser.write('hello'.encode())
time.sleep(1)
