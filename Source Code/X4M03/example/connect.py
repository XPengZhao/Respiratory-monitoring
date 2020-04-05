import pymoduleconnector
import time

device_name = "COM3"

mc = pymoduleconnector.ModuleConnector(device_name)
xep = mc.get_xep()
print("receive ping, the value is:", hex(xep.ping()))

xep.module_reset()
mc.close()
time.sleep(3)