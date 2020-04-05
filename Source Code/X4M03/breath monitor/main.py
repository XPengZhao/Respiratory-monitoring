import numpy as np
from time import sleep

import pymoduleconnector
from pymoduleconnector import DataType

from xep_data import Xep_data
import processing

device_name = 'COM3'
FPS = 17
iterations = 16
pulses_per_step = 300
dac_min = 949
dac_max = 1100
area_start = 0.4
area_end = 5.0

sample_time = 60

amp_datamatrix = np.loadtxt('./data/amp_matrix223sample_time60s.txt')
pha_datamatrix = np.loadtxt('./data/pha_matrix223sample_time60s.txt')

if __name__ == '__main__':

    # my_xep1 = Xep_data(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end)
    # my_xep1.display_sys_info()
    # a,b = my_xep1.get_data_matrix(sample_time,save = True)
    # my_xep1.plot_frame(amp_datamatrix, pha_datamatrix, sample_time)
  
    # location = processing.find_peak(amp_datamatrix[50])
    # print(location)

    # processing.FFT_fasttime(amp_datamatrix)
    processing.lowpass_filter(amp_datamatrix)
