#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

c = 0.1
D = 1.7


def four_para_logisical(theta, a, b, gamma=1):
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
    p = c + (gamma - c) / (1 + np.exp(-D * a * (theta - b)))
    return p


def likehood(theta, a, b, y, gamma=1):
    p = four_para_logisical(theta, a, b, gamma=gamma)
    l = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    # print 'title', D * a * (theta - b), p
    return l


def step_reduce(theta, y, ab0, step, gamma=1):
    like0 = likehood(theta, ab0[(0, 0)], ab0[(0, 1)], y, gamma=gamma)
    # print 'like0', like0
    ab1 = ab0 - step
    # print 'ab0', ab0
    # print 'ab1', ab1
    like1 = likehood(theta, ab1[(0, 0)], ab1[(0, 1)], y, gamma=gamma)
    # print 'like1', like1
    # print 'step', step
    while True:
        if like1 < like0:
            step = step / 2.0
            ab1 = ab0 - step
            like1 = likehood(theta, ab1[(0, 0)], ab1[(0, 1)], y, gamma=gamma)
            # print 'like..', like1
            # print step
        else:
            break

    return step


def newton(theta, y, gamma=1, eps=1e-6, mxite=30, lbd=1e-4):

    ab = np.mat([1.0, 0.0])
    err = 10.0
    iters = 0
    while err >= eps and iters < mxite:
        # print '-' * 30
        iters += 1
        # print 'ab', ab
        a = ab[(0, 0)]
        b = ab[(0, 1)]
        p = four_para_logisical(theta, a, b, gamma=gamma)
        # ai偏导
        fa = D / (1 - c) * np.sum((theta - b) * (p - c) * (y - p) / p)
        # bi偏导
        fb = -D * a / (1 - c) * np.sum((y - p) * (p - c) / p)
        # ai bi导
        fab = -D / (1 - c) * \
              np.sum((p - c) * ((y / p - 1) + D * a / (1 - c) * (theta - b) * (1 - p) / p * (y * c / p - p)))
        print '一阶导', fa, fb

        # ai二阶导
        ffa = (D / (1 - c)) ** 2 * np.sum((theta - b) ** 2 * (p - c) * (1 - p) / p * (y * c / p - p))
        # bi二阶导
        ffb = (D * a / (1 - c)) ** 2 * np.sum((p - c) * (1 - p) / p * (y * c / p - p))

        print '二阶导', fab, ffa, ffb

        hessian = np.mat(([ffa, fab], [fab, ffb]))  # 2 * 2
        gradient = np.mat([fa, fb])  # 1 * 2
        dia = np.identity(np.shape(hessian)[0]) * lbd
        hessian = hessian + dia
        inverse = hessian.I
        print 'hessian矩阵'
        print hessian
        step = np.dot(inverse, gradient.T)
        step = step_reduce(theta, y, ab, step.T, gamma=gamma)

        err = np.sum(abs(step)).mean()
        print '-' * 40

        # print likehood(theta, a, b, y, gamma=gamma)
        # ab = ab - step.T
        ab = ab - step
        # print likehood(theta, ab[(0, 0)], ab[(0, 1)], y, gamma=gamma)
        print '-' * 40
        # print step.T
        # print ab
        # print '- ' * 40
        print 'step.t', step.T
        print 'new step', step
        print 'err', err
        print 'ab', ab
    return err, iters, a, b


def draw(theta, b, label, gamma):

    plt.figure()
    a = np.arange(-3, 3, 0.1)
    lk = [likehood(theta, i, b, label, gamma=gamma) for i in a]
    # 求p
    # lk = [four_para_logisical(theta, i, b, gamma=gamma)[0][0] for i in a]
    plt.plot(a, lk)
    print a[np.array(lk).argmax()]
    plt.show()


if __name__ == '__main__':
    # data = np.random.random((10, 1))
    # label = np.random.randint(2, size=(10, 1))
    # print np.shape(data)
    # print np.shape(label)
    data = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [0.8]])
    # data = np.array([[0.1]])
    # label = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1]])
    label = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0]])
    # print data
    # print label
    print 'data', data

    print 'label', label
    print np.shape(data), np.shape(label)
    error, iters, a, b = newton(data, label, eps=1e-10, mxite=100)
    print 'err, iters, a, b', error, iters, a, b

    draw(data, b, label, gamma=1)
