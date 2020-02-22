#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" \example X4M200_sleep_record.py

#Target module: X4M200

#Introduction: This is an example of how to record respiration and movement data for sleep monitoring from X4M200.

#Command to run: "python X4M200_sleep_record.py -d com8" or "python3 X4M200_sleep_record.py -d com8"
                change "com8" with your device name, using "--help" to see other options.
                Using TCP server address as device name is also supported, e.g. 
                "python X4M200_sleep_record.py -d tcp://192.168.1.169:3000".
"""

from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optparse import OptionParser
import numpy as np
from time import sleep
from pymoduleconnector import ModuleConnector
from pymoduleconnector import DataType
from pymoduleconnector import DataRecorder
from pymoduleconnector import RecordingOptions
import pymoduleconnector


def on_file_available(data_type, filename):
    print("new file available for data type: {}".format(data_type))
    print("  |- file: {}".format(filename))
    if data_type == DataType.BasebandApDataType:
        print("processing baseband ap data from file")
    elif data_type == DataType.SleepDataType:
        print("processing sleep data from file")


def on_meta_file_available(session_id, meta_filename):
    print("new meta file available for recording with id: {}".format(session_id))
    print("  |- file: {}".format(meta_filename))


def record_sleep (device_name):
    directory = "."

    # Reset module
    mc = ModuleConnector(device_name)
    x4m200 = mc.get_x4m200()
    x4m200.module_reset()
    mc.close()
    sleep(3)

    # Assume an X4M300/X4M200 module and try to enter XEP mode
    mc = ModuleConnector(device_name)
    x4m200 = mc.get_x4m200()
    recorder = mc.get_data_recorder()

    recorder = mc.get_data_recorder()
    recorder.subscribe_to_file_available(pymoduleconnector.AllDataTypes, on_file_available)
    recorder.subscribe_to_meta_file_available(on_meta_file_available)
    recorder.start_recording(DataType.BasebandApDataType | DataType.SleepDataType, directory)

    # Stop running application and set module in manual mode.
    try:
        x4m200.set_sensor_mode(0x01, 0) # Make sure no profile is running
    except RuntimeError:
        # Profile not running, OK
        pass

    # Load x4m200 respiration detection profile
    x4m200.load_profile(0x47fabeba)
    try:
        x4m200.set_sensor_mode(0x01,0)#RUN mode
    except RuntimeError:
        # Sensor already stopped, OK
        pass
    x4m200.set_output_control(0x610a3b00,1)
    while True:
        rdata = x4m200.read_message_respiration_sleep()
        print ("Frame: {} RPM: {} Distance: {} Movement Slow: {} Movement Fast: {}" .format(rdata.frame_counter, rdata.respiration_rate, rdata.distance, rdata.movement_slow, rdata.movement_fast))
        sleep(0.2)


def main():
    parser = OptionParser()
    parser.add_option(
        "-d",
        "--device",
        dest = "device_name",
        help = "device file to use",
        metavar = "FILE")

    (options, args) = parser.parse_args()

    if not options.device_name:
        print ("you have to specify device, e.g.: python record.py -d /dev/ttyACM0")
        sys.exit(1)

    record_sleep(options.device_name)


if __name__ == "__main__":
    main()