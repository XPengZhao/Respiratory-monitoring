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
area_end = 5

sample_time = 70
slice_time_start = 10
slice_time_end = sample_time

env_amp_datamatrix = np.loadtxt('./data/datamatrix_env_sampletime70s/amp_matrix.txt')
fro_amp_datamatrix = np.loadtxt('./data/datamatrix_front_sampletime70s/amp_matrix.txt')

amp_datamatrix = np.loadtxt('./data/amp_matrix4627sampletime65s.txt')
pha_datamatrix = np.loadtxt('./data/pha_matrix4627sampletime65s.txt')

if __name__ == '__main__':

    # my_xep1 = Xep_data(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end)
    # my_xep1.display_sys_info()
    # a,b = my_xep1.get_data_matrix(sample_time,save = True)
    # my_xep1.plot_frame(a, b, sample_time)
    # print(a.shape)
  
    # location = processing.find_peak(amp_datamatrix[50])
    # print(location)

    # processing.FFT_fasttime(amp_datamatrix)
    # processing.lowpass_filter(fro_amp_datamatrix)

    env_amp_datamatrix = processing.slice_datamatrix(env_amp_datamatrix, slice_time_start, slice_time_end, FPS)
    fro_amp_datamatrix = processing.slice_datamatrix(fro_amp_datamatrix, slice_time_start, slice_time_end, FPS)
    sub_amp_datamatrix = fro_amp_datamatrix - env_amp_datamatrix
    sample_time = sample_time - slice_time_start

    processing.plot_frame_animation(sub_amp_datamatrix, sample_time, FPS, area_start, area_end)
    # processing.plot_amp_datamatrix3D(fro_amp_datamatrix, sample_time, FPS, area_start, area_end)
    # processing.plot_slowtime_profile(sub_amp_datamatrix, 36)
    # processing.lowpass_filter(sub_amp_datamatrix, 36, FPS, sample_time)



