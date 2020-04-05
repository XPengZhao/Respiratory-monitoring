from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# 找到波形的峰值
def find_peak(array_1D):
    doublediff = np.diff(np.sign(np.diff(array_1D)))
    peak_locations = np.where(doublediff == -2)[0] + 1
    return peak_locations

def FFT_fasttime(amp_datamatrix):
    line_need = amp_datamatrix[ 85: , 25]
    # fd_line = np.fft.fft(line_need)

    fig = plt.figure()
    fig.suptitle("one frame")
    plt.plot(line_need)
    plt.show()

def lowpass_filter(amp_datamatrix):
    line_need = amp_datamatrix[ 170: , 24]

    b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    filtedData = signal.filtfilt(b, a, line_need)  #data为要过滤的信号
    c, d = signal.butter(8, 0.02, 'highpass')
    filtedData = signal.filtfilt(c, d, filtedData)

    
    fd_data = np.fft.fft(filtedData)


    fig = plt.figure()
    fig.suptitle("low pass")
    raw_signal = fig.add_subplot(3,1,1)
    lowpass_signal = fig.add_subplot(3,1,2)
    fft_signal = fig.add_subplot(3,1,3)

    raw_signal.set_title("raw signal")
    raw_signal.set_xlabel('Range/m')
    lowpass_signal.set_title("lowpass_signal")
    fft_signal.set_title("fft_signal")

    line1, = raw_signal.plot(line_need)
    line2, = lowpass_signal.plot(filtedData)
    line3, = fft_signal.plot(fd_data)
    
    plt.show()
