# -*- coding: utf-8 -*-
""" \example X4M200_X4M300_printout_pulsedoppler_data.py

#Target module: X4M200,X4M300

#Introduction:  This is an example of how to set up a module for streaming pulse-Doppler
            telegrams, and how to assemble them to whole range-Doppler matrices.
            
#Command to run:  "python X4M200_X4M300_printout_pulsedoppler_data.py -d com8" or "python3 X4M200_X4M300_printout_pulsedoppler_data.py -d com8"
                change "com8" with your device name, using "--help" to see other options. 
                Using TCP server address as device name is also supported, e.g. 
                "python X4M200_sleep_record.py -d tcp://192.168.1.169:3000".
"""

from __future__ import print_function

import numpy as np
import pymoduleconnector
from pymoduleconnector.ids import *
from optparse import OptionParser
import sys
import time

parser = OptionParser()
parser.add_option(
    "-d",
    "--device",
    dest="device_name",
    help="device file to use, example: python %s -d COM4"%sys.argv[0],
    metavar="FILE")

parser.add_option(
    "-i",
    "--interface",
    dest="interface",
    default="x4m300",
    help="Interface to use. x4m300 or x4m200",
    metavar="IF")

parser.add_option(
    "-w",
    "--datatype",
    dest="datatype",
    default="pulsedoppler",
    help="Data to get. pulsedoppler or noisemap",
    metavar="TYPE")

parser.add_option(
    "-f",
    "--format",
    dest="format",
    default="byte",
    help="Data format to get. byte or float",
    metavar="FORMAT")

parser.add_option(
    "-m",
    "--dopplers",
    dest="dopplers",
    default="both",
    help="Which pD instance to get. fast, slow or both.",
    metavar="FORMAT")

parser.add_option(
    "-n",
    "--num-messages",
    dest="num_messages",
    type=int,
    default=0,
    help="how many matrices to read (0 = infinite)",
    metavar="INT")

(options, args) = parser.parse_args()

if not options.device_name:
    print("Please specify a device name, example: python %s -d COM4"%sys.argv[0])
    sys.exit(1)

mc = pymoduleconnector.ModuleConnector(options.device_name, 0)
time.sleep(1)

if options.interface == "x4m300":
    app = mc.get_x4m300()
    app.ping()
    app.load_profile(XTS_ID_APP_PRESENCE_2)
elif options.interface == "x4m200":
    app = mc.get_x4m200()
    app.ping()
    app.load_profile(XTS_ID_APP_RESPIRATION_2)
else:
    print("Interface not recognized.", file=sys.stderr)
    raise SystemExit(1)

# Flush all buffers
while app.peek_message_pulsedoppler_byte():
    app.read_message_pulsedoppler_byte()
while app.peek_message_pulsedoppler_float():
    app.read_message_pulsedoppler_float()
while app.peek_message_noisemap_byte():
    app.read_message_noisemap_byte()
while app.peek_message_noisemap_float():
    app.read_message_noisemap_float()

# Turn on outputs.
if options.dopplers == "both":
    ctrl = XTID_OUTPUT_CONTROL_PD_FAST_ENABLE | XTID_OUTPUT_CONTROL_PD_SLOW_ENABLE
elif options.dopplers == "fast":
    ctrl = XTID_OUTPUT_CONTROL_PD_FAST_ENABLE
elif options.dopplers == "slow":
    ctrl = XTID_OUTPUT_CONTROL_PD_SLOW_ENABLE
else:
    print("Pulse-Doppler instance not recognized.", file=sys.stderr)
    raise SystemExit(1)

if options.datatype == "pulsedoppler" and options.format == "byte":
    app.set_output_control(XTS_ID_PULSEDOPPLER_BYTE, ctrl)
elif options.datatype == "pulsedoppler" and options.format == "float":
    app.set_output_control(XTS_ID_PULSEDOPPLER_FLOAT, ctrl)
elif options.datatype == "noisemap" and options.format == "byte":
    app.set_output_control(XTS_ID_NOISEMAP_BYTE, ctrl)
elif options.datatype == "noisemap" and options.format == "float":
    app.set_output_control(XTS_ID_NOISEMAP_FLOAT, ctrl)
else:
    print("Datatype/format not recognized.", file=sys.stderr)
    raise SystemExit(1)

# Start sensor
app.set_sensor_mode(XTS_SM_RUN, 0)

# Set up state machinery for matrix assembly
matrix = []
is_negative = True
prev_range_idx = -1
ok = True
instance = None
n_printed = 0
got_errors = False

# Assemble whole range-Doppler matrices from the individual packets
while True:
    if options.datatype == "pulsedoppler" and options.format == "byte":
        pkt = app.read_message_pulsedoppler_byte()
    elif options.datatype == "pulsedoppler" and options.format == "float":
        pkt = app.read_message_pulsedoppler_float()
    elif options.datatype == "noisemap" and options.format == "byte":
        pkt = app.read_message_noisemap_byte()
    elif options.datatype == "noisemap" and options.format == "float":
        pkt = app.read_message_noisemap_float()

    if instance is None:
        instance = pkt.pulsedoppler_instance

    if pkt.range_idx == 0 and is_negative and matrix:
        # We now have a complete range-Doppler matrix.
        matrix = np.array(matrix)
        print("Instance:", ["slow", "fast"][instance], matrix.shape)
        print(matrix)
        n_printed += 1
        if options.num_messages != 0 and n_printed >= options.num_messages:
            break
        # Reset
        matrix = []
        prev_range_idx = -1
        instance = pkt.pulsedoppler_instance

    if is_negative and pkt.range_idx == prev_range_idx + 1:
        ok = True
        prev_range_idx = pkt.range_idx
    elif not is_negative and pkt.range_idx == prev_range_idx:
        pass
    else:
        if ok:
            print("bad range_idx ", pkt.range_idx, " in instance ", instance,
                    ", prev ", prev_range_idx ,", resetting and waiting for "
                    "zero.", sep="")
            matrix = []
            is_negative = True
            prev_range_idx = -1
            ok = False
            got_errors = True
        continue

    if options.format == "byte":
        # Convert byte data to float.
        data = bytearray(pkt.get_data())
        offset = pkt.byte_step_start
        step = pkt.byte_step_size
        data = [10.0**((float(x)*step + offset)/10.0) for x in data]
    else:
        data = list(pkt.get_data())

    # A noisemap misses a piece in the middle, so fill it with zeroes
    # Has no effect if nothing is missing.
    if is_negative:
        pkt_end = pkt.frequency_start + pkt.frequency_step * pkt.frequency_count
        if pkt_end < 0.000001:
            data.extend(0 for _ in range(int(round(-pkt_end / pkt.frequency_step))))
    else:
        if pkt.frequency_start > 0.000001:
            pre = [0 for _ in range(int(round(pkt.frequency_start / pkt.frequency_step)))]
            data = pre + data

    if len(matrix) <= pkt.range_idx:
        matrix.append([])

    matrix[pkt.range_idx].extend(data)

    is_negative ^= True

app.set_sensor_mode(XTS_SM_STOP, 0)

if got_errors:
    sys.exit(1)