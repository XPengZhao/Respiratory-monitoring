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

env_amp_datamatrix = np.loadtxt('./data/exp8/datamatrix_env_sampletime70s/amp_matrix.txt')
obj_amp_datamatrix = np.loadtxt('./data/exp8/datamatrix_apart_sampletime70s/amp_matrix.txt')


if __name__ == '__main__':

    # sleep(5)
    # my_xep1 = Xep_data(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end)
    # my_xep1.display_sys_info()
    # a,b = my_xep1.get_data_matrix(sample_time,save = True)
    # my_xep1.plot_frame(a, b, sample_time)
    # print(a.shape)

    obj_amp_datamatrix = processing.slice_datamatrix(obj_amp_datamatrix, slice_time_start, slice_time_end, FPS)
    env_amp_datamatrix = processing.slice_datamatrix(env_amp_datamatrix, slice_time_start, slice_time_end, FPS)
    obj_amp_datamatrix[:,:4] = 0
    env_amp_datamatrix[:,:4] = 0
    obj_dif_datamatrix = processing.get_diffmatrix(obj_amp_datamatrix)
    sub_amp_datamatrix = obj_amp_datamatrix - env_amp_datamatrix
    # sub_amp_datamatrix[:,50:60] = 0
    med_obj_datamatrix = processing.get_median_matrix(sub_amp_datamatrix)
    std_obj_datamatrix = processing.get_stdmatrix(obj_amp_datamatrix)
    sub_avr_datamatrix = processing.get_avrmatrix(sub_amp_datamatrix)
    env_avr_datamatrix = processing.get_avrmatrix(env_amp_datamatrix)
    obj_avr_datamatrix = processing.get_avrmatrix(obj_amp_datamatrix)
    sub_min_datamatrix = processing.get_minmatrix(sub_amp_datamatrix)
    sample_time = sample_time - slice_time_start



    # processing.plot_frame_animation(obj_amp_datamatrix, sample_time, FPS, area_start, area_end)
    # processing.plot_peak_animation(med_obj_datamatrix, sample_time, FPS, area_start, area_end,True)
    # processing.plot_amp_datamatrix3D(sub_amp_datamatrix, sample_time, FPS, area_start, area_end)
    # processing.plot_slowtime_profile(sub_amp_datamatrix,20,35,False)
    # processing.lowpass_filter(obj_amp_datamatrix, 32, FPS, sample_time)
    # processing.find_peakline(sub_amp_datamatrix)
    # processing.plot_heatmap(obj_amp_datamatrix)
    # processing.plot_datamatrix3D_multicolor(sub_amp_datamatrix, sample_time, FPS, area_start, area_end)
    # processing.subtract_env(med_obj_datamatrix, FPS)
    # processing.plot_std(std_obj_datamatrix)
    # processing.plot_avr(env_avr_datamatrix,obj_avr_datamatrix,sub_avr_datamatrix)
    processing.plot_multi(obj_amp_datamatrix,12,27,False)
    # processing.breathmonitor_realtime(obj_amp_datamatrix, 32, FPS, sample_time)




