from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

title_font = {'fontname':'Arial', 'size':16, 'color':'black', 'weight':'bold'}
axis_font = {'family':'Times New Roman', 'weight':'normal', 'size':14}

def slice_datamatrix(amp_datamatrix, start, end, FPS):
    return amp_datamatrix[ start*FPS:end*FPS ,:]

def plot_heatmap(amp_datamatrix):

    extent_matrix = np.zeros((1020,8))
    amp_datamatrix = np.hstack((extent_matrix,amp_datamatrix))

    #将横纵坐标都映射到（0，1）的范围内  
    extent=(0,1,0,1) 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Data Matrix Heatmap", title_font)
    ax.set_xlabel("Time/s", axis_font)
    ax.set_ylabel("Range/m", axis_font)
    ax.set_xticks(np.linspace(0,1,7))
    ax.set_xticklabels( ('0','10','20','30','40','50','60'))
    ax.set_yticks(np.linspace(0,1,6)) 
    ax.set_yticklabels( ('0', '1', '2', '3', '4','5'))

    matrix = amp_datamatrix.T
    matrix = matrix[::-1]        #上下翻转
    im = ax.imshow(matrix,extent=extent,cmap='rainbow')
    plt.colorbar(im)
    plt.show()
    # fig.savefig('heatmap.png',dpi=600)

def plot_amp_datamatrix3D(amp_datamatrix, sampletime, FPS, area_start, area_end):
    """
    功能：绘制数据矩阵
    """

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

def plot_datamatrix3D_multicolor(amp_datamatrix, sampletime, FPS, area_start, area_end):

    bin_length = 8*1.5e8/23.328e9

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title("Raw radar data", title_font)
    ax.set_xlabel("Range/m", axis_font)
    ax.set_ylabel("Time/s", axis_font)
    ax.set_zlabel("Amplitude", axis_font)
    ax.set_zlim(-0.005, 0.005)
    ax.set_xlim(0,5)
    ax.set_ylim(0, 60)

    slowtime = np.arange(sampletime)
    fasttime = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)
    norm = plt.Normalize(amp_datamatrix.min(), amp_datamatrix.max())
    
    for i in slowtime:
        slowtime_array = np.full(len(fasttime), i)
        amp_array = amp_datamatrix[i*FPS]
        points = np.array([fasttime,slowtime_array,amp_array]).T.reshape(-1,1,3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # ax.plot3D(fasttime ,slowtime_array,amp_datamatrix[i*FPS] ,'blue')
        lc = Line3DCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(amp_array)
        lc.set_linewidth(1)
        line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    plt.show()
    fig.savefig('test.png',dpi=600)


def plot_frame_animation(amp_matrix, sampletime, FPS, area_start, area_end):
    """
    功能：绘制雷达帧动图
    """

    bin_length = 8*1.5e8/23.328e9
    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)
    amp_fig.set_ylim(0, 0.015)

    amp_fig.set_title("Radar Frame")
    amp_fig.set_xlabel("Range/m")
    amp_fig.set_ylabel("Amplitude")
    line1, = amp_fig.plot(ax_x, amp_matrix[0])
    sc = amp_fig.scatter(ax_x,amp_matrix[0],s=16)

    def animate(i):
        fig.suptitle("frame count:%d" % i)
        amplitude = amp_matrix[i]
        line1.set_ydata(amplitude)
        sc.set_offsets(np.c_[ax_x, amp_matrix[i]])
        return line1,

    ani = FuncAnimation(fig, animate, frames = sampletime*FPS, interval=1/FPS*1000)
    # ani.save('./radar_frame.gif',writer='imagemagick',fps=17)
    plt.show()


def get_stdmatrix(amp_matrix):
    """
    功能：求信号标准差
    """

    std_matrix = np.zeros(91)
    col = len(amp_matrix[0])

    for i in range(col):
        std_matrix[i] = np.std(amp_matrix[:,i])

    return std_matrix

def get_diffmatrix(amp_matrix):
    """
    功能：获得一阶时间差分矩阵（变化率）
    """
    diffmatrix = np.zeros((1020,91))
    for i in range(1019):
        diffmatrix[i] = amp_matrix[i+1] -amp_matrix[i]
    diffmatrix[1019] = 0
    return diffmatrix

def get_avrmatrix(amp_matrix):
    """
    功能：计算平均值矩阵
    """
    col = len(amp_matrix[0])
    avrmatrix = np.zeros(91)

    for i in range(col):
        avrmatrix[i] = np.sum(amp_matrix[ : ,i])/1020

    return avrmatrix

def get_median_matrix(amp_matrix):
    """
    对慢时间序列进行中位数滤波
    """
    med_matrix = np.zeros((1020,91))
    col = len(amp_matrix[0])

    for i in range(col):
        med_matrix[:,i] = signal.medfilt(amp_matrix[:,i],11)
    # for i in range(col):
    #     med_matrix[:,i] = avrfilter(med_matrix[:,i],3)
    # for i in range(col):
    #     med_matrix[:,i] = signal.medfilt(med_matrix[:,i],7)
    # for i in range(col):
    #     med_matrix[:,i] = signal.medfilt(med_matrix[:,i],11)

    return med_matrix


def avrfilter(array,length):
    col =len(array)
    n = int((length-1)/2)
    result = np.zeros(col)

    for i in range(n):
        result[i] = np.sum(array[:i+n+1])/(i+n+1)

    for i in range(n,col-n):
        result[i] = np.sum(array[i-n:i+n+1])/length

    for i in range(col-n,col):
        result[i] = np.sum(array[i-n:])/(col-i+n)

    return result

def plot_peak_animation(amp_datamatrix, sampletime, FPS, area_start, area_end,filtered = False):
    """
    功能：绘制峰值动图
    """
    fs = 17
    N = fs*sampletime
    # N = fs*20
    deri_datamatrix = get_diffmatrix(amp_datamatrix)    #差分
    index_matrix = []
    peakarea_matrix = np.zeros((1020,13))
    peakdiff_matrix = np.zeros((1020,13))
    index = np.where(amp_datamatrix[0]==np.max(amp_datamatrix[0]))[0][0]
    peakarea_matrix[0] = amp_datamatrix[0][index-6:index+7]
    peakdiff_matrix[0] = deri_datamatrix[0][index-6:index+7]
    index_matrix.append(index)

    for i in range(1, 1020):
        slice_matrix = amp_datamatrix[i][index-6:index+7]
        index = np.where(amp_datamatrix[i]==np.max(slice_matrix))[0][0]
        peakarea_matrix[i] = amp_datamatrix[i][index-6:index+7]
        peakdiff_matrix[i] = deri_datamatrix[i][index-6:index+7]
        index_matrix.append(index)

    # """
    # 调制波
    # """
    # derimatrix = np.sign(derimatrix)    #符号函数 1 0 -1
    # sqr_wave = derimatrix[:,0] + derimatrix[:,4]
    # sqr_wave[np.where(sqr_wave == 2)[0]] = 1
    # sqr_wave[np.where(sqr_wave == -2)[0]] = -1

    # diff_matrix = peakarea_matrix[ : , 6:7]
    # diff_matrix = np.median(peakarea_matrix,axis=1)
    # print(diff_matrix.shape)


    diff_matrix = peakarea_matrix[ : , 4:9]
    diff_matrix = diff_matrix
    diff_matrix = np.sum(diff_matrix,axis=1)


    med_matrix = signal.medfilt(diff_matrix,1)
    avr_matrix = avrfilter(med_matrix,1)


    if filtered:
        b, a = signal.butter(8, 0.06, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
        c, d = signal.butter(8, 0.015, 'highpass')
        filtedData = signal.filtfilt(b, a, avr_matrix)  #data为要过滤的信号
        filtered_matrix = signal.filtfilt(c, d, filtedData)

    # filtered_matrix = np.multiply(filtered_matrix,np.hamming(60*FPS))
    fd_data = np.fft.rfft(filtered_matrix)

    doublediff = np.diff(np.sign(np.diff(filtered_matrix)))
    peak_locations = np.where(doublediff == -2)[0] + 1
    print(peak_locations.shape)

    # fd_data = win_fft(filtered_matrix,20,6)

    bin_length = 8*1.5e8/23.328e9
    # ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)
    ax_x = np.arange(1020)/17
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1)*60, fs/N*60)

    fig = plt.figure()
    amp_fig = fig.add_subplot(2,1,1)
    fft_fig = fig.add_subplot(2,1,2)
    # amp_fig.set_ylim(0, 0.015)

    amp_fig.set_title("Slow time profile")
    amp_fig.set_xlabel("Time/s")
    amp_fig.set_ylabel("Amplitude")
    fft_fig.set_xlabel("Time/s")
    fft_fig.set_xlim(0,50)


    line1, = amp_fig.plot(ax_x, med_matrix,color='red')
    line3, = amp_fig.plot(ax_x, avr_matrix,color='green')
    line2, = amp_fig.plot(ax_x, filtered_matrix)
    line4, = fft_fig.plot(ax_x2, abs(fd_data)*2/N)


    # line1, = amp_fig.plot(ax_x, amp_datamatrix[0])
    # sc = amp_fig.scatter(index_matrix[0]*bin_length+0.4,amp_datamatrix[0][index_matrix[0]])

    # def animate(i):
    #     fig.suptitle("frame count:%d" % i)
    #     amplitude = amp_datamatrix[i]
    #     line1.set_ydata(amplitude)
    #     sc.set_offsets((index_matrix[i]*bin_length+0.4, amp_datamatrix[i][index_matrix[i]]))
    #     return line1,

    # ani = FuncAnimation(fig, animate, frames = sampletime*FPS, interval=1/FPS*1000)
    # ani.save('./radar_frame.gif',writer='imagemagick',fps=17)
    plt.show()
    fig.savefig('test.png',dpi=600)



def win_fft(matrix_1d,length,num):

    FPS = 17

    filtered_matrix = np.multiply(matrix_1d[:length*FPS],np.hamming(length*FPS))
    fd_data = np.fft.rfft(filtered_matrix)

    for i in range(1,num-1):
        start = int(i/2*length*FPS)
        stop = int((i/2+1)*length*FPS)
        filtered_matrix = np.multiply(matrix_1d[start:stop],np.hamming(length*FPS))
        fd_data += np.fft.rfft(filtered_matrix)

    return fd_data





def plot_slowtime_profile(amp_datamatrix, bin_start, bin_end, filtered=True):
    """
    功能：绘制从bin_start到bin_end条Range Bin的慢时间采样图
    """

    bin_range = np.arange(bin_start, bin_end)
    num = len(bin_range)

    sum_matrix = np.zeros(1020)
    fig = plt.figure()
    raw_signal = fig.add_subplot(1, 1, 1)
    offset = 0.0010
    for i in range(num):
        rangebin_data = amp_datamatrix[ : , bin_range[i]]
        if filtered:
            b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
            c, d = signal.butter(8, 0.015, 'highpass')
            filtedData = signal.filtfilt(b, a, rangebin_data)  #data为要过滤的信号
            rangebin_data = signal.filtfilt(c, d, filtedData)
        sum_matrix += rangebin_data

        # raw_signal.set_title("Slow time profile", title_font)
        # raw_signal.set_xlabel('Time/s', axis_font)
        # raw_signal.set_ylabel('Amplitude', axis_font)
        ax_x = np.arange(0, 60, 1/17)
        line1, = raw_signal.plot(ax_x, rangebin_data+offset*i)

    plt.show()



def find_peakline(amp_datamatrix):
    """
    功能：跟踪目标范围段雷达的峰值，绘制图像
    """
    peakline_matrix = np.zeros(1020)
    peakline_matrix[0] = np.max(amp_datamatrix[0])
    index = np.where(amp_datamatrix[0]==np.max(amp_datamatrix[0]))[0][0]
    
    for i in range(1, 1020):
        # print(index)
        slice_matrix = amp_datamatrix[i][index-1:index+2]
        peakline_matrix[i] = np.max(slice_matrix)
        index = np.where(amp_datamatrix[i]==np.max(slice_matrix))[0][0]

    b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    c, d = signal.butter(8, 0.015, 'highpass')
    filtedData = signal.filtfilt(b, a, peakline_matrix)  #data为要过滤的信号
    peakline_matrix = signal.filtfilt(c, d, filtedData)

    fig = plt.figure()
    raw_signal = fig.add_subplot(1,1,1)
    ax_x = np.arange(0, 60, 1/17)
    line1, = raw_signal.plot(ax_x, peakline_matrix)

    plt.show()




def FFT_fasttime(amp_datamatrix, bin_num, FPS, sampletime):
    win_length = 30
    rangebin_data = amp_datamatrix[ : , 36]

    sig_win = np.multiply(rangebin_data[ 170:win_length*FPS+170], np.hamming(win_length*FPS))
    fd_data = np.fft.rfft(sig_win)

    fig = plt.figure()
    fft_signal = fig.add_subplot(1,1,1)

    fs = FPS
    N = win_length*FPS
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1), fs/N)

    fft_signal.set_title("FFT signal",title_font)
    fft_signal.set_xlabel("Freq/Hz", axis_font)
    fft_signal.set_ylabel("Amplitude", axis_font)
    
    line3, = fft_signal.plot(ax_x2, abs(fd_data)*2/N)
    plt.show()



def lowpass_filter(amp_datamatrix, bin_num, FPS, sampletime):
    rangebin_data = amp_datamatrix[ : , bin_num]

    b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    filtedData = signal.filtfilt(b, a, rangebin_data)  #data为要过滤的信号
    c, d = signal.butter(8, 0.01, 'highpass')
    filtedData = signal.filtfilt(c, d, filtedData)

    fd_data = np.fft.rfft(filtedData)

    fig = plt.figure()
    raw_signal = fig.add_subplot(3,1,1)
    bandpass_signal = fig.add_subplot(3,1,2)
    fft_signal = fig.add_subplot(3,1,3)

    fs = FPS
    N = fs*sampletime
    ax_x = np.arange(0, 60, 1/17)
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1)*60, fs/N*60)

    raw_signal.set_title("Raw signal", title_font)
    raw_signal.set_xlabel("Time/s", axis_font)
    raw_signal.set_ylabel("Amplitude", axis_font)

    bandpass_signal.set_title("BPF signal",title_font)
    bandpass_signal.set_xlabel("Time/s", axis_font)
    bandpass_signal.set_ylabel("Amplitude", axis_font)

    fft_signal.set_title("FFT signal",title_font)
    fft_signal.set_xlabel("BPM", axis_font)
    fft_signal.set_ylabel("Amplitude", axis_font)
    fft_signal.set_xlim(0,50)

    line1, = raw_signal.plot(ax_x, rangebin_data)
    line2, = bandpass_signal.plot(ax_x, filtedData)
    line3, = fft_signal.plot(ax_x2, abs(fd_data)*2/N)

    fd_data = abs(fd_data)*2/N
    print(np.where(fd_data == np.max(fd_data))[0][0])

    plt.show()
    # fig.savefig('test.png',dpi=600)



def subtract_env(amp_datamatrix,FPS):

    """
    功能：检测雷达帧中有运动的部分
    """

    detect_matrix = amp_datamatrix[0:54,:]
    row_num = detect_matrix.shape[0]
    col_num = detect_matrix.shape[1]
    sub_matrix = np.empty(col_num)
    threshold = 1e-4
    area_start = 0.4
    area_end = 5
    bin_length = 8*1.5e8/23.328e9

    for i in range(col_num):
        sub_matrix[i] = np.max(detect_matrix[:,i]) - np.min(detect_matrix[:,i])
    sub_matrix = np.where(sub_matrix>threshold)[0]*bin_length + area_start
    print(sub_matrix)


    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)
    amp_fig.set_ylim(0, 0.01)
    amp_fig.set_title("Radar Frame",title_font)
    amp_fig.set_xlabel("Range/m",axis_font)
    amp_fig.set_ylabel("Amplitude",axis_font)

    for i in range(row_num):
        line = amp_fig.plot(ax_x, detect_matrix[i],linewidth=0.8)
    # plt.vlines(1.63, 0, 0.01, colors = "r", linestyles = "dashed")
    # plt.vlines(2.04, 0, 0.01, colors = "r", linestyles = "dashed")
    # plt.vlines(3.02, 0, 0.01, colors = "b", linestyles = "dashed")
    # plt.vlines(3.12, 0, 0.01, colors = "b", linestyles = "dashed")
    # plt.vlines(3.58, 0, 0.01, colors = "g", linestyles = "dashed")
    # plt.vlines(3.74, 0, 0.01, colors = "g", linestyles = "dashed")
    # amp_fig.plot(ax_x, amp_datamatrix[0])


    plt.show()
    
    # fig.savefig('test.png',dpi=600)


def get_minmatrix(datamatrix):
    minmatrix = np.zeros(91)
    for i in range(91):
        minmatrix[i] = np.min(datamatrix[:,i])
    return minmatrix

def plot_std(std_datamatrix):
    area_start = 0.4
    area_end = 5
    bin_length = 8*1.5e8/23.328e9
    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    threshold_matrix = np.zeros(91)
    max_std = np.max(std_datamatrix)
    index = np.where(std_datamatrix == max_std)[0][0]
    d_0 = index*bin_length+0.4    # 最高方差点距离
    # p_m = np.sum(std_datamatrix[index-1:index+2])/3
    p_m = max_std
    n = np.mean(std_datamatrix)
    sigma_0 = (p_m)/(2*d_0)
    for i in range(91):
        d_i = i*bin_length+0.4
        k = np.square(d_i)/np.square(d_0)
        sigma_i = 1/k * sigma_0
        threshold_matrix[i] = sigma_i

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)
    amp_fig.set_ylim(0, 0.001)
    amp_fig.set_title("Standard Deviation",title_font)
    amp_fig.set_xlabel("Range/m",axis_font)
    amp_fig.set_ylabel("Amplitude",axis_font)

    amp_fig.plot(ax_x, std_datamatrix,linewidth=1)
    # amp_fig.scatter(d_0,sigma_0,s=16)
    amp_fig.plot(ax_x, threshold_matrix,linewidth=1.5)
    plt.show()
    # fig.savefig('std.png',dpi=600)

def plot_avr(env_avrmatrix,obj_avrmatrix,sub_avrmatrix):
    area_start = 0.4
    area_end = 5
    bin_length = 8*1.5e8/23.328e9
    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)

    amp_fig.set_title("Standard Deviation",title_font)
    amp_fig.set_xlabel("Range/m",axis_font)
    amp_fig.set_ylabel("Amplitude",axis_font)
    amp_fig.set_ylim(0, 0.01)
    amp_fig.plot(ax_x, env_avrmatrix,linewidth=1,color='r')
    amp_fig.plot(ax_x, obj_avrmatrix,linewidth=1,color='g')
    amp_fig.plot(ax_x, sub_avrmatrix,linewidth=1,color='b')
    plt.show()
    fig.savefig('avr.png',dpi=600)



def plot_multi(amp_datamatrix,line1,line2,filtered = False):
    """
    功能：绘制从bin_start到bin_end条Range Bin的慢时间采样图
    """

    area_start = 0.4
    area_end = 5
    bin_length = 8*1.5e8/23.328e9
    fs = 17
    N = fs*60

    fig = plt.figure()
    raw_signal = fig.add_subplot(1, 1, 1)

    data1 = amp_datamatrix[ : , line1]
    data2 = amp_datamatrix[ : , line2]
    # data3 = amp_datamatrix[ : , 22]

    d_1 = line1*bin_length + 0.4
    d_2 = line2*bin_length + 0.4
    k = np.square(d_2)/np.square(d_1)
    data2_i = data2 - 1/k * data1
    # data2 = data2 - 0.1*data1
    # data1 = data1 - 5*data2

    if filtered:
        b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
        c, d = signal.butter(8, 0.015, 'highpass')
        filtedData = signal.filtfilt(b, a, data1)  #data为要过滤的信号
        data1 = signal.filtfilt(c, d, filtedData)
        filtedData = signal.filtfilt(b, a, data2_i)  #data为要过滤的信号
        data2_i = signal.filtfilt(c, d, filtedData)


    raw_signal.set_title("Slow time profile", title_font)
    raw_signal.set_xlabel('Time/s', axis_font)
    raw_signal.set_ylabel('Amplitude', axis_font)
    # raw_signal.set_ylim(0,0.009)
    raw_signal.set_xlim(0,60)

    fd_data = np.fft.rfft(data2_i)
    ax_x = np.arange(1020)/17
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1)*60, fs/N*60)

    ax_x = np.arange(0, 60, 1/17)
    line1, = raw_signal.plot(ax_x, data2,color='red')
    line2, = raw_signal.plot(ax_x, data2_i, color = 'blue')
    line3, = raw_signal.plot(ax_x,1/k * data1,color = 'green')
    # line3, = raw_signal.plot(ax_x2, abs(fd_data)*2/N, color = 'green')

    plt.legend(handles=[line1,line2],labels=['object1','object2'],loc='upper right')
    plt.show()
    fig.savefig('test.png',dpi=600)


def breathmonitor_realtime(amp_datamatrix, bin_num, FPS, sampletime):
    """
    实时呼吸监测
    """
    rangebin_data = amp_datamatrix[ : , bin_num]
    area_start = 0.4
    area_end = 5
    breathrate = []
    time_bin = 1/17
    

    b, a = signal.butter(8, 0.08, 'lowpass')   #配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    filtedData = signal.filtfilt(b, a, rangebin_data)  #data为要过滤的信号
    c, d = signal.butter(8, 0.01, 'highpass')
    filtedData = signal.filtfilt(c, d, filtedData)

    for i in range(30):
        filtered_matrix = filtedData[i*17:(i+30)*17]
        doublediff = np.diff(np.sign(np.diff(filtered_matrix)))
        peak_locations = np.where(doublediff == -2)[0] + 1
        breathrate.append(peak_locations.shape[0]*2)

    bin_length = 8*1.5e8/23.328e9
    ax_x = np.arange((area_start-1e-5), (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(1,1,1)
    amp_fig.set_ylim(0, 0.015)


    amp_fig.set_xlabel("Range/m")
    amp_fig.set_ylabel("Amplitude")
    line1, = amp_fig.plot(ax_x, amp_datamatrix[0])
    plt.vlines(1.84, 0, 0.015, colors = "r", linestyles = "dashed")

    def animate(i):

        count = i // 17
        breath_now = breathrate[count]
        amp_fig.set_title("Time = %.2fs       Breath rate = %d" % (30+i*time_bin, breath_now),title_font)
        amplitude = amp_datamatrix[i]
        line1.set_ydata(amplitude)
        return line1,

    ani = FuncAnimation(fig, animate, frames = 30*FPS, interval=1/FPS*1000)
    ani.save('./radar_frame.gif',writer='imagemagick',fps=17)
    plt.show()


