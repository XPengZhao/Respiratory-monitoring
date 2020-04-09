from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

title_font = {'fontname':'Arial', 'size':18, 'color':'black', 'weight':'bold'}
axis_font = {'family':'Times New Roman', 'weight':'normal', 'size':16}

def slice_datamatrix(amp_datamatrix, start, end, FPS):
    return amp_datamatrix[ start*FPS:end*FPS ,:]




def plot_amp_datamatrix3D(amp_datamatrix, sampletime, FPS, area_start, area_end):
    bin_length = 8*1.5e8/23.328e9

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title("Raw radar data", title_font)
    ax.set_xlabel("Range/m", axis_font)
    ax.set_ylabel("Time/s", axis_font)
    ax.set_zlabel("Amplitude", axis_font)
    ax.set_zlim(-0.015, 0.015)
    ax.set_xlim(0,5)

    slowtime = np.arange(sampletime)
    fasttime = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)
    
    for i in slowtime:
        slowtime_array = np.full(len(fasttime), i)
        ax.plot3D(fasttime ,slowtime_array,amp_datamatrix[i*FPS] ,'blue')
    plt.show()




def plot_frame_animation(amp_matrix, sampletime, FPS, area_start, area_end):

    bin_length = 8*1.5e8/23.328e9
    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)
    amp_fig.set_ylim(0, 0.015)

    amp_fig.set_title("Radar Frame")
    amp_fig.set_xlabel("Range/m")
    amp_fig.set_ylabel("Amplitude")
    line1, = amp_fig.plot(ax_x, amp_matrix[0])

    def animate(i):
        fig.suptitle("frame count:%d" % i)
        amplitude = amp_matrix[i]
        line1.set_ydata(amplitude)
        return line1,

    ani = FuncAnimation(fig, animate, frames = sampletime*FPS, interval=1/FPS*1000)
    # ani.save('./radar_frame.gif',writer='imagemagick')
    plt.show()

def plot_slowtime_profile(amp_datamatrix, bin_num):
    rangebin_data = amp_datamatrix[ : , bin_num]

    fig = plt.figure()
    raw_signal = fig.add_subplot(1,1,1)
    raw_signal.set_title("Slow time profile", title_font)
    raw_signal.set_xlabel('Time/s', axis_font)
    raw_signal.set_ylabel('Amplitude', axis_font)
    ax_x = np.arange(0, 60, 1/17)
    line1, = raw_signal.plot(ax_x, rangebin_data)

    plt.show()



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




def lowpass_filter(amp_datamatrix, bin_num, FPS, sampletime):
    rangebin_data = amp_datamatrix[ : , bin_num]

    b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    filtedData = signal.filtfilt(b, a, rangebin_data)  #data为要过滤的信号
    c, d = signal.butter(8, 0.02, 'highpass')
    filtedData = signal.filtfilt(c, d, filtedData)

    fd_data = np.fft.rfft(filtedData)

    fig = plt.figure()
    raw_signal = fig.add_subplot(3,1,1)
    bandpass_signal = fig.add_subplot(3,1,2)
    fft_signal = fig.add_subplot(3,1,3)

    fs = FPS
    N = fs*sampletime
    ax_x = np.arange(0, 60, 1/17)
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1), fs/N)

    raw_signal.set_title("Raw signal", title_font)
    raw_signal.set_xlabel("Time/s", axis_font)
    raw_signal.set_ylabel("Amplitude", axis_font)

    bandpass_signal.set_title("BPF signal",title_font)
    bandpass_signal.set_xlabel("Time/s", axis_font)
    bandpass_signal.set_ylabel("Amplitude", axis_font)

    fft_signal.set_title("FFT signal",title_font)
    fft_signal.set_xlabel("Freq/Hz", axis_font)
    fft_signal.set_ylabel("Amplitude", axis_font)

    line1, = raw_signal.plot(ax_x, rangebin_data)
    line2, = bandpass_signal.plot(ax_x, filtedData)
    line3, = fft_signal.plot(ax_x2, abs(fd_data)*2/N)

    plt.show()



def bandpass_filter(amp_datamatrix):
    pass