# -*- coding: utf-8 -*-

# Connects to an X2M200 sensor, sends a ping command and prints the response
# Methods throw a RuntimeException if they fail.

from pymoduleconnector import ModuleConnector
device_name = 'COM3'
mc = ModuleConnector(device_name, log_level=0)
x4m200 = mc.get_x4m200()
# x4m200.set_sensor_mode_idle()
pong = x4m200.ping()
print('Receive Pong, the value is', hex(pong))