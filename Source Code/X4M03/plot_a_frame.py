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

def get_config_info(mc):
    xep = mc.get_xep()
    # print("The DAC iteration is", xep.x4driver_get_iterations(), "\n")
    # print(xep.x4driver_get_dac_min())
    print("The FPS is %.1f \n" % xep.x4driver_get_fps())
    print("the Frame area is from %.1f to %.1f:" % (xep.x4driver_get_frame_area().start, \
        xep.x4driver_get_frame_area().end))
    # print("get decimation factor %d" % xep.get_decimation_factor())



if __name__ == "__main__":

    FPS = 10
    reset(my_device)
    mc = pymoduleconnector.ModuleConnector(my_device)

    xep = mc.get_xep()
    print(xep.get_system_info(2))

    # Set DAC range
    xep.x4driver_set_dac_min(900)
    xep.x4driver_set_dac_max(1150)

    # Set integration
    xep.x4driver_set_iterations(16)
    xep.x4driver_set_pulses_per_step(26)

    # Start streaming of data
    xep.x4driver_set_fps(FPS)

    get_config_info(mc)

    #read a frame
    d = xep.read_message_data_float()
    frame = np.array(d.data)
    fd_frame = np.fft.fft(frame)

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("one frame")
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_ylim(-0.003,0.003) #keep graph in frame (FIT TO YOUR DATA)
    # ax1.set_xlim(300,400)

    line, = ax1.plot(frame)
    ax2.plot(fd_frame)
    plt.show()

    np.savetxt("1.txt", frame)


    clear_buffer(mc)