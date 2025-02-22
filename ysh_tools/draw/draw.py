import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from mpl_toolkits.mplot3d import Axes3D

def read_data_from_file(file_path, num=-1):
    '''
    if num==-1, then no constraint to the number of datas
    file format:
        1 2 3.2 4
        2 3 4.3 5
        ...
    '''
    datas = []
    with open(file_path, 'r') as file:  #打开文件
        file_data = file.readlines() #读取所有行
        column_num = len(file_data[0].split(' '))
        for i in range(column_num):
            datas.append([])
        for i, row in enumerate(file_data):
            if row == '\n':
                continue
            if num>=0 and i>=num:
                break
            tmp_list = row.split(' ') #按‘，’切分每行的数据
            tmp_list[-1] = tmp_list[-1].replace('\n','') #去掉换行符
            for j in range(column_num):
                datas[j].append(float(tmp_list[j]))
    
    return column_num, datas

def draw():
    x = [1, 3, 5, 7, 9]
    y = [2, 6, 4, 8, 10]
    z = [7, 8, 9]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
    fig.subplots_adjust(hspace=0.4, wspace=0.4) # 用来调整 matplotlib 中子图之间的间距

    # 散点图
    axes[0][0].scatter(x, y, c='red', s=30, marker='D', alpha = 1/2, label='Data')
    # 标题和 xy 坐标轴的标题
    axes[0][0].set_title('Title', fontsize=10)
    axes[0][0].set_xlabel('xlabel', fontsize=10,fontfamily = 'sans-serif',fontstyle='italic')
    axes[0][0].set_ylabel('ylabel', fontsize='x-large', fontstyle='oblique')
    axes[0][0].legend()            # 显示图例，但这里没有为散点图指定标签
    # xy 坐标轴的一些属性设定
    axes[0][0].set_aspect('equal') # 设置子图的纵横比为相等，确保 X 轴和 Y 轴的单位长度是相同的
    axes[0][0].minorticks_on()     # 打开次刻度线，显示主要刻度之间的次要刻度
    axes[0][0].set_xlim(0,10)      # 设置 X 轴的显示范围为 [0, 16]
    axes[0][0].set_ylim(0,10)      # 设置 y 轴的显示范围为 [0, 16]
    axes[0][0].grid(which='minor', axis='both')    # 在子图上为次刻度线绘制网格，which='minor' 表示网格线应用于次刻度，axis='both' 表示网格线应用于 X 轴和 Y 轴
    # 坐标轴tick和细节
    axes[0][0].xaxis.set_tick_params(rotation=45, labelsize=8, colors='b') # 设置 X 轴刻度的参数
    start, end = axes[0][0].get_xlim() # 获取当前 X 轴的显示范围（即 X 轴的最小值和最大值）
    axes[0][0].xaxis.set_ticks(np.arange(start, end, 1)) 
    axes[0][0].yaxis.tick_right()

    # 添加 colorbar 
    # [Matplotlib 系列：colorbar 的设置](https://blog.csdn.net/weixin_43257735/article/details/121831188)
    cmap = copy.copy(mpl.cm.viridis)                # 颜色的映射规则
    norm = mpl.colors.Normalize(vmin=0, vmax=10)    # 颜色归一化到归一化 [0, 10] 区间内
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    axes[0][1].scatter(x, y, c=[data for data in y], cmap=cmap, norm=norm)
    fig.colorbar(mappable=mappable, ax=axes[0][1], orientation='vertical', ticks=np.linspace(0, 10, 11))  # 可以把颜色图展示在旁边

    # line
    axes[1][0].plot(x, y, '*:r', markersize=5, linewidth=1)    # 格式化字符串 fmt 格式: marker|linestyle|color

    # 柱状图
    axes[1][1].bar(x, y, width=1)

    # others: 直方图 饼图

    plt.show()

def draw3D():
    x = [1, 3, 5]
    y = [2, 4, 6]
    z = [7, 8, 9]

    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    ax1.scatter(x, y, c='red', alpha = 1/10)

    # 散点图
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter3D(x, y, z, c='red', alpha = 1/2)
    
    # 曲面
    ax3 = fig.add_subplot(223, projection='3d')
    x_d = np.arange(0, 10, 0.1)
    y_d = np.arange(0, 10, 0.1)
    x_grid, y_grid = np.meshgrid(x_d, y_d)
    z_grid = x_grid + y_grid + x_grid*y_grid
    ax3.plot_surface(x_grid, y_grid, z_grid ,cmap='rainbow')   # 绘制曲面

    plt.show()

def from_file(file_path):
    x = []
    x0 = []
    x1 = []
    y_area = []
    y_delay = []
    with open(file_path, 'r') as file:  #打开文件
        file_data = file.readlines() #读取所有行
        for i, row in enumerate(file_data):
            tmp_list = row.split(' ') #按‘，’切分每行的数据
            tmp_list[-1] = tmp_list[-1].replace('\n','') #去掉换行符
            x.append([int(tmp_list[0]), int(tmp_list[1])])
            x0.append(int(tmp_list[0]))
            x1.append(int(tmp_list[1]))
            y_area.append(float(tmp_list[2]))
            y_delay.append(float(tmp_list[3]))
            if i>10000:
                break

    x = np.array(x)
    y_area = np.array(y_area)
    y_delay= np.array(y_delay)
    y = y_delay + y_area
    x0_d = np.arange(190, 200, 0.1)
    x1_d = np.arange(40, 60, 0.1)
    x0_grid, x1_grid = np.meshgrid(x0_d, x1_d)
    y_grid = 5851.62 - 60.99*x0_grid + 8.970*x1_grid + 0.1621*x0_grid*x0_grid - 0.0360*x0_grid*x1_grid - 0.0127*x1_grid*x1_grid
    
    #定义坐标轴
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

    ax.scatter3D(x0, x1, y, c='red')  # 绘制散点图
    ax.plot_surface(x0_grid, x1_grid, y_grid ,cmap='rainbow')   # 绘制曲面
    plt.show()

def draw_feature(file_path):
    top_num = 10
    x = []
    for j in range(top_num):
        x.append([])
    y = []
    with open(file_path, 'r') as file:  #打开文件
        file_data = file.readlines() #读取所有行
        for i, row in enumerate(file_data):
            feature = [0] * 8
            tmp_list = row.split(' ') #按‘，’切分每行的数据
            tmp_list[-1] = tmp_list[-1].replace('\n','') #去掉换行符
            feature[:3] = [int(v) for v in tmp_list[:3]]
            ct32_column_top_n = []
            ct22_column_top_n = []
            ct32_ct22_column_top_n = []
            for j, v in enumerate(tmp_list[3:13]):
                if j == 0:
                    v = v.replace('[', '')
                elif j == top_num-1:
                    v = v.replace(']', '')
                ct32_column_top_n.append(int(v))
            for j, v in enumerate(tmp_list[13:23]):
                if j == 0:
                    v = v.replace('[', '')
                elif j == top_num-1:
                    v = v.replace(']', '')
                ct22_column_top_n.append(int(v))
            for j, v in enumerate(tmp_list[23:33]):
                if j == 0:
                    v = v.replace('[', '')
                elif j == top_num-1:
                    v = v.replace(']', '')
                ct32_ct22_column_top_n.append(int(v))
            feature[3] = ct32_column_top_n
            feature[4] = ct22_column_top_n
            feature[5] = ct32_ct22_column_top_n
            feature[6] = tmp_list[33]
            feature[7] = tmp_list[34]
            feature[6:] = [float(v) for v in tmp_list[33:]]
            # print('NO. {}: {}'.format(i, feature))

            for j in range(top_num):
                x[j].append(feature[3][j]) # ct32 top10
            y.append(feature[7]) # delay
            if i>10000:
                break

    x = np.array(x)
    y = np.array(y)  
    
    fig, axes = plt.subplots(nrows=3, ncols=4)
    axes[0][0].scatter(x[0], y, c='red', alpha = 1/10)
    axes[0][1].scatter(x[1], y, c='red', alpha = 1/10)
    axes[0][2].scatter(x[2], y, c='red', alpha = 1/10)
    axes[0][3].scatter(x[3], y, c='red', alpha = 1/10)
    axes[1][0].scatter(x[4], y, c='red', alpha = 1/10)
    axes[1][1].scatter(x[5], y, c='red', alpha = 1/10)
    axes[1][2].scatter(x[6], y, c='red', alpha = 1/10)
    axes[1][3].scatter(x[7], y, c='red', alpha = 1/10)
    axes[2][0].scatter(x[8], y, c='red', alpha = 1/10)
    axes[2][1].scatter(x[9], y, c='red', alpha = 1/10)
    
    axes[0][0].set_title('top 10')
    axes[0][1].set_title('top 9')
    axes[0][2].set_title('top 8')
    axes[0][3].set_title('top 7')
    axes[1][0].set_title('top 6')
    axes[1][1].set_title('top 5')
    axes[1][2].set_title('top 4')
    axes[1][3].set_title('top 3')
    axes[2][0].set_title('top 2')
    axes[2][1].set_title('top 1')

    plt.show()


def main():
    # 输入数据
    # X: x1 x2
    x = np.array([[197, 51], [198, 48], [198, 47], [198, 46], [197, 47], [197, 48], [196, 49], [194, 49], [195, 49], [195, 48], [195, 48], [196, 47], [197, 46], [198, 45], [197, 46], [197, 46], [196, 47], [198, 45], [197, 46], [198, 45]])
    y_area = np.array([1953.0 , 1938.0 , 1941.75, 1930.25, 1936.5 , 1943.75, 1936.5 , 1929.5 , 1941.25, 1938.25, 1942.25, 1939.75, 1937.25, 1937.25, 1934.0 , 1930.75, 1931.0 , 1926.5 , 1926.0 , 1921.25])
    y_delay= np.array([1.24285,  1.268949,  1.249525,  1.276425,  1.256175,  1.2335 ,  1.245099,  1.25925,  1.225475,  1.225475,  1.212925,  1.215775,  1.217175,  1.211075,  1.2167 ,  1.219075,  1.218  ,  1.224825,  1.220424,  1.218025])  
    y = 0.1*y_area + y_delay*100
    x0 = x[:, 0]
    x1 = x[:, 1]

    x0_d = np.arange(190, 200, 0.1)
    x1_d = np.arange(40, 60, 0.1)
    x0_grid, x1_grid = np.meshgrid(x0_d, x1_d)

    # y_grid = 429.5 + 68.75*x0_grid - 228.8*x1_grid - 0.3122*x0_grid*x0_grid + 1.16*x0_grid*x1_grid + 0.042*x1_grid*x1_grid
    # y_grid = 1367 + 1.934*x0_grid + 3.998*x1_grid
    y_grid = 17140 - 189.7*x0_grid + 67.54*x1_grid + 0.5075*x0_grid*x0_grid - 0.1674*x0_grid*x1_grid - 0.3476*x1_grid*x1_grid

    #定义坐标轴
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

    ax.scatter3D(x0,x1,y, c='red')  # 绘制散点图
    # ax.plot3D(x0_d, x1_d ,y_d ,'gray')    # 绘制空间曲线
    ax.plot_surface(x0_grid, x1_grid, y_grid ,cmap='rainbow')   # 绘制曲面
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()
    return

    z = np.linspace(0,13,1000)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    zd = 13*np.random.random(100)
    xd = 5*np.sin(zd)
    yd = 5*np.cos(zd)

    plt.scatter(X, y, c='red')
    plt.plot(X, model.predict(X_poly), c='green')
    plt.show()


if __name__ == '__main__':
    print('===== START =====')
    # main()
    # area_delay()
    # area_delay_from_file('get3.txt')
    # from_file('get4.txt')
    draw_feature('get5.txt')
    print('=====  END  =====')