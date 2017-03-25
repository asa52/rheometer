import serial
import time

start = time.time()
print(start)
ser = serial.Serial('COM4', 115200, timeout=1)
time.sleep(1)
print("hello")
data = []
while time.time() - start < 10:
	data.append(ser.readline())

print("time's up!")
print (data)
ser.close()