import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == "__main__":

    FPS = 10
    reset(my_device)
    mc = pymoduleconnector.ModuleConnector(my_device)

    xep = mc.get_xep()
    # Set DAC range
    xep.x4driver_set_dac_min(900)
    xep.x4driver_set_dac_max(1150)

    # Set integration
    xep.x4driver_set_iterations(16)
    xep.x4driver_set_pulses_per_step(26)

    # Start streaming of data
    xep.x4driver_set_fps(FPS)
    print("the FPS is:", xep.x4driver_get_fps())
    print("the Frame area is:", xep.x4driver_get_frame_area().end)
    #read a frame
    d = xep.read_message_data_float()
    frame = np.array(d.data)

    fc = 7.29e9 # Lower pulse generator setting
    fs = 23.328e9 # X4 sampling rate
    csine = np.exp(-1j*fc/fs*2*np.pi*np.arange(len(frame)))
    cframe = frame * csine

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("one frame")
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_ylim(-0.003,0.003) #keep graph in frame (FIT TO YOUR DATA)
    # ax2.set_xlim(0,100)

    line, = ax1.plot(frame)
    ax2.plot(cframe)
    plt.show()

    clear_buffer(mc)