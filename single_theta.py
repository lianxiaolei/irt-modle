#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

c = 0.1
D = 1.7


def four_para_logisical(x, theta, gamma=1):
    '''
    三参logistic模型
    :param c:  学生蒙对概率
    :param gammai:  学生想多了概率
    :param a:  试题区分度
    :param b:  试题难度
    :param theta:   学生能力
    :param D:   参数
    :return:    学生答对试题的概率
    '''
    p = c + (gamma - c) / (1 + np.exp(-D * x[:, 0] * (theta - x[:, 1])))
    return p


def newton(x, y, gamma=1, eps=1e-6, mxite=30):
    theta = 1
    a = x[:, 0]
    err = 10
    iters = 0
    while err >= eps and iters < mxite:
        iters += 1
        p = four_para_logisical(x, theta, gamma=gamma)
        # 一阶导
        f = np.sum(a * (y - p) * (p - c) / (p * (1 - c))) * D
        # 二阶导
        ff = np.sum(a ** 2 * (p - c) * (y * c - p ** 2) * (1 - p) / (p ** 2 * (1 - c) ** 2)) * D ** 2
        print('f, ff', f, ff)
        if f == 0:
            if ff < 0:
                return err, iters, theta
        if ff == 0:
            theta = theta - eps
            err = eps
            continue
        # 牛顿法
        theta = theta - f / ff
        err = abs(f / ff)

    return err, iters, theta


def likehood(theta, x, y, gamma=1):
    p = four_para_logisical(x, theta, gamma=gamma)

    l = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return l


if __name__ == '__main__':
    data = np.random.random((10, 2))
    label = np.random.randint(2, size=10)
    # print data
    # print label

    err, iters, theta = newton(data, label, eps=1e-30)
    print('err, iters, t', err, iters, theta)

    import matplotlib.pyplot as plt

    plt.figure()
    xx = np.arange(-3, 3, 0.01)
    yy = []
    yy = [likehood(i, data, label, gamma=1) for i in xx]
    plt.plot(xx, yy)
    print(xx[np.array(yy).argmax()])
    plt.show()
