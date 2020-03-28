import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep

import pymoduleconnector
from pymoduleconnector import DataType

my_device = "COM3"


def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    sleep(3)

def clear_buffer(mc):
    """Clears the frame buffer"""
    xep = mc.get_xep()
    while xep.peek_message_data_float():
        xep.read_message_data_float()

def display_sys_info(xep):
    print("FirmWareID =", xep.get_system_info(2), "\n")
    print("Version =", xep.get_system_info(3), "\n")
    print("Build =", xep.get_system_info(4), "\n")
    # print("SerialNumber =", xep.get_system_info(6), "\n")
    print("VersionList =", xep.get_system_info(7), "\n")

def read_apdata(xep):
    #read a frame
    data = xep.read_message_data_float().data
    data_length = len(data)

    i_vec = np.array(data[:data_length//2])
    q_vec = np.array(data[data_length//2:])
    iq_vec = i_vec + 1j*q_vec

    ph_ampli = abs(iq_vec)                       #振幅
    ph_phase = np.arctan2(q_vec, i_vec)          #相位

    return ph_ampli, ph_phase

if __name__ == "__main__":

    FPS = 17
    reset(my_device)
    mc = pymoduleconnector.ModuleConnector(my_device)

    xep = mc.get_xep()
    display_sys_info(xep)

    xep.x4driver_init()
    xep.x4driver_set_downconversion(1)


    xep.x4driver_set_iterations(16)
    xep.x4driver_set_pulses_per_step(300)
    xep.x4driver_set_dac_min(949)
    xep.x4driver_set_dac_max(1100)

    # Set frame area offset
    xep.x4driver_set_frame_area_offset(0.18)

    # Set frame area
    xep.x4driver_set_frame_area(0.4, 5.0)
    start = xep.x4driver_get_frame_area().start
    end = xep.x4driver_get_frame_area().end

    # Start streaming of data
    xep.x4driver_set_fps(FPS)

    bin_length = 8 * 1.5e8/23.328e9  # range_decimation_factor * (c/2) / fs.
    ax_x = np.arange((start-1e-5), (end-1e-5)+bin_length, bin_length)

    amplitude, phase = read_apdata(xep)

    fig = plt.figure()
    fig.suptitle("frame")
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)


    ax1.set_ylim(0,0.01) #keep graph in frame (FIT TO YOUR DATA)
    # ax2.set_xlim(0,100)
    
    line1, = ax1.plot(ax_x,amplitude)
    line2, = ax2.plot(ax_x,phase)

    clear_buffer(mc)

    def animate(i):
        amplitude, phase = read_apdata(xep)
        line1.set_ydata(amplitude)
        line2.set_ydata(phase)
        return line1, line2,

    ani = FuncAnimation(fig, animate, interval=FPS)

    try:
        plt.show()
    finally:
        # Stop streaming of data
        xep.x4driver_set_fps(0)

