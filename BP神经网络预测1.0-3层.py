# -*- coding = utf-8 -*-
# @Software : PyCharm
# 使用者:dzy
# 开发时间:2021/12/1 11:26
import xlrd
from numpy import random,dot,exp,array
import numpy as np
import traceback
import time
def bpnn(yxys_n,sj_n,l1_nn,l2_nn,eta,dd_n,jd,random_seed):
    print('正在运行，请稍等，大约需要20秒-5分钟不等，速度由电脑配置决定')
    def maxminnorm(array):
        global maxcols, mincols
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        maxcols = array.max(axis=1).reshape(data_rows, 1)
        mincols = array.min(axis=1).reshape(data_rows, 1)
        t = np.empty((data_rows, data_cols))
        for i in range(data_rows):
            t[i, :] = (array[i, :] - mincols[i]) / (maxcols[i] - mincols[i])
        return t

    def maxminnorm2(array):
        global maxcols2, mincols2
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        maxcols2 = array.max(axis=1).reshape(data_rows, 1)
        mincols2 = array.min(axis=1).reshape(data_rows, 1)
        t = np.empty((data_rows, data_cols))
        for i in range(data_rows):
            t[i, :] = (array[i, :] - mincols2[i]) / (maxcols2[i] - mincols2[i])
        return t

    def remaxminnorm(array):
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        t = np.empty((data_rows, data_cols))
        for i in range(data_rows):
            t[i, :] = ((array[i, :]) * (maxcols2[i] - mincols2[i]) + mincols2[i])
        return t

    def usemaxminnorm(array):
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        t = np.empty((data_rows, data_cols))
        for i in range(data_rows):
            t[i, :] = ((array[i, :] - mincols[i]) / (maxcols[i] - mincols[i]))
        return t

    def relu(x):
        return np.maximum(0.01 * x, x)

    def fp(input):
        l1 = relu(dot(input, w0))
        l2 = relu(dot(l1, w1))
        l3 = relu(dot(l2, w2))
        return l1, l2, l3

    def bp(l1, l2, l3, y,eta):
        global l2_error
        # 计算原本的结果和预测的结果误差
        eta = eta
        l2_error = y - l3                       # 实际值-预测值
        l2_slpoe = np.where(l3 > 0, 1, 0.01)
        l2_delta = eta * l2_error * l2_slpoe
        l1_error = l2_delta.dot(w2.T)  # 实际值与预测结果的误差，
        l1_slope = np.where(l2 > 0, 1, 0.01)  # 计算斜率  (为了梯度下降法
        l1_delta = eta * l1_error * l1_slope  # 调整值    调整w1
        l0_slope = np.where(l1 > 0, 1, 0.01)  # 斜率
        l0_error = l1_delta.dot(w1.T)  # 第一层神经元算出的结果 与 第二层神经元间的误差，体现为第一层神经元的权重变化
        l0_delta = eta * l0_slope * l0_error  # 计算w0的误差  调整值    ,调整w0
        # 计算增量
        return l0_delta, l1_delta, l2_delta
    # 数据处理
    table = xlrd.open_workbook(r"./数据收集.xlsx", 'rb')
    sheet1 = table.sheet_names()[0]
    sheet1 = table.sheet_by_index(0)
    rowNum = sheet1.nrows
    colNum = sheet1.ncols
    newlist = []
    for i in range(rowNum):                 # 创建空列表，储存数据
        rowi = sheet1.row_values(i)
        newlist.append(rowi[2:])
    # 引入数据
    ys_arr_empty = np.empty([yxys_n, sj_n])    # 创建一个空array  用于储存影响因素
    for i in range(yxys_n):
        ys_arr_empty[i] = newlist[i + 2][3:]
        gjys = ys_arr_empty
    gjys = maxminnorm(gjys).T
    gj = array(newlist[1][3:]).reshape(1,sj_n)
    gj = maxminnorm2(gj).T  # 注释掉，则尝试不标准化结果的预测。  若使用：取消掉上面的转置，加到此处
    # 3.设置随机权重
    random.seed(random_seed)
    w0 = random.random((yxys_n,l1_nn)) * 2 - 1
    w1 = random.random((l1_nn,l2_nn)) * 2 - 1
    w2 = random.random((l2_nn,1)) * 2 - 1
    # # 4.循环
    i = 0
    c = 0
    for it in range(dd_n):
        l0 = gjys
        l1,l2,l3 = fp(l0)
        l0_delta,l1_delta,l2_delta = bp(l1,l2,l3,gj,eta)
        w2 = w2 + dot(l2.T, l2_delta)
        w1 = w1 + dot(l1.T, l1_delta)
        w0 = w0 + dot(l0.T, l0_delta)
        i+=1
        a = dot(l2_error.T,l2_error)          #计算误差
        # print(i)                      # 使用print（i） 观察迭代次数
        # print(a)                      # 使用print（a） 观察方差
        if abs(c-a)<jd :          # 两次误差调整幅度小于 0.00001时 结束循环
            print('--------------------------------')
            print('迭代次数：',i)
            print('本次误差：',a)
            print('上次误差：',c)                    # 上次的误差值
            break
        elif abs(c-a)>=jd:
            c = a
        if i == dd_n:
            print('----------------------------------')
            print('已迭代',i,'次，超出设置的迭代次数')
    gjys_y_arry = np.empty([yxys_n,4])                      # 创建用于存储预测数据的空array
    for i in range(yxys_n):
        gjys_y_arry[i] = newlist[i + 2][:4]
        gjys_y = gjys_y_arry                                # 填入数据
    gjys_y = usemaxminnorm(gjys_y).T
    # gj_y = fp(gjys_y)[2]
    gj_y = remaxminnorm(fp(gjys_y)[2].T).T   # 注释掉，则尝试不标准化结果的预测。  若使用：取消掉上面的转置，加到此处，同时下面的显示改为gj_y
    print('--------------------------------')
    print(' 预测结果：',gj_y[0],'\n',
          '第一个检测值：',gj_y[1],'\n',
          '第二个检测值：',gj_y[2],'\n',
          '第三个检测值（用于对比训练值）：',gj_y[3])

while True:
    try:
        while True:
            yxys_n, sj_n = int(input('请输入影响因素个数(敲击回车确定):')), int(input('请输入样本个数：'))
            print('输入成功:', yxys_n, sj_n)
            l1_nn, l2_nn = int(input('请输入第一层神经元个数(建议在"影响因素个数/2"附近浮动）)：')), int(input('请输入第二层神经元个数：'))
            print('输入成功:', l1_nn, l2_nn)
            dd_n, jd = int(input('请输入迭代次数(整数):')), float(input('请输入精度，即当两次迭代间的误差小于此数时，停止迭代：'))
            print('输入成功:', dd_n, jd)
            eta = float(input('请输入eta（学习率）值：'))
            print('输入成功:', eta)
            random_seed = int(input('请设置随机种子（整数，相同的随机种子，在数据完全相同的情况下，可以迭代出相同的结果）:'))
            print('输入成功：', random_seed)
            bpnn(yxys_n, sj_n, l1_nn, l2_nn, eta, dd_n, jd, random_seed)
            print('---------------------------------------')
            while True:
                a_continue = str(input('是否继续？（y/n）:'))
                if a_continue == 'y':
                    break
                elif a_continue == 'n':
                    break
                else:
                    print('输入错误,请输入 y 或 n')
                    continue
            if a_continue == 'y':
                continue
            elif a_continue == 'n':
                break
        break
    except Exception:
        print('-------------------------------')
        traceback.print_exc()
        time.sleep(1)
        print('输入有误,请重新输入')
        print('-------------------------------')


