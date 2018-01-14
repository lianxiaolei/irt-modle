#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rd

reload(sys)
sys.setdefaultencoding('utf8')

c = 0.25
D = 1.7


def three_para_logisical(theta, a, b, gamma=1):
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
    '''
    似然函数
    :param theta:   学生能力
    :param a:   试题区分度
    :param b:   试题难度
    :param y:   学生答题结果
    :param gamma:   学生想多了概率
    :return:    似然值
    '''
    p = three_para_logisical(theta, a, b, gamma=gamma)

    if np.any(1 <= p):
        print '异常'
        print 'a', a
        print 'b', b
        print '1p', p
    if np.any(0 >= p):
        print '异常'
        print '0p', p

    l = np.sum(y * np.log(p) + (1 - y) * np.log(1.0 - p))

    return l


def newton_for_theta(a, b, y, origin_t=1, gamma=1, eps=1e-6, mxite=30, lbd=0.001, low_bound=-3.0, up_bound=3.0):
    '''
    牛顿法求theta
    :param theta:
    :param a:
    :param b:
    :param y:
    :param gamma:
    :param eps:
    :param mxite:
    :return:
    '''
    err = 10.0
    iters = 0

    lkhd = -1e3
    result_theta = None
    for nums in range(0, 20):
        theta = origin_t
        while err >= eps and iters < mxite:
            iters += 1
            p = three_para_logisical(theta, a, b, gamma=gamma)
            # 一阶导
            f = np.sum(a * (y - p) * (p - c) / (p * (1 - c))) * D
            # 二阶导
            ff = np.sum(a ** 2 * (p - c) * (y * c - p ** 2) * (1 - p) / (p ** 2 * (1 - c) ** 2)) * D ** 2
            # print 'f, ff', f, ff
            # print 'f/ff', f/ff
            if f == 0:
                if ff < 0:
                    return err, iters, theta
            if ff == 0:
                ff = ff - lbd
                # theta = theta - eps
                # theta = theta - f / ff
                # # err = eps
                # err = f / ff
                # continue
            # 牛顿法
            step = f / ff
            step = step_reduce_theta(theta, y, a, b, step, gamma=gamma, low_bound=low_bound, up_bound=up_bound)

            theta = theta - step
            theta = min(up_bound, max(low_bound, theta))
            # print 'lkhd', likehood(theta, a, b, y, gamma=gamma)
            err = abs(step)
        new_lkhd = likehood(theta, a, b, y, gamma=gamma)
        if new_lkhd > lkhd:
            lkhd = new_lkhd
            result_theta = theta

    return err, iters, result_theta
    # return theta


def step_reduce_theta(theta0, y, a, b, step, gamma=1, low_bound=-3.0, up_bound=3.0):
    '''
    下山法
    :param theta:
    :param y:
    :param ab0:
    :param step:
    :param gamma:
    :return:
    '''
    like0 = likehood(theta0, a, b, y, gamma=gamma)
    theta = theta0 - step
    theta = min(up_bound, max(low_bound, theta))
    like1 = likehood(theta, a, b, y, gamma=gamma)
    while like1 < like0:
        step = step / 2.0
        theta = theta0 - step
        theta = min(up_bound, max(low_bound, theta))
        like1 = likehood(theta, a, b, y, gamma=gamma)
    return step


def newton_for_ab(theta, y, origin_a=1, origin_b=0, gamma=1, eps=1e-6, mxite=30, lbd=1e-4, low_bound=np.array([-3.0, -3.0]),
                  up_bound=np.array([3.0, 3.0])):
    '''
    牛顿法求ab
    :param theta:
    :param y:
    :param gamma:
    :param eps:
    :param mxite:
    :param lbd:
    :return:
    '''

    err = 1.0
    iters = 0

    lkhd = -1e3
    result_ab = None
    for nums in range(0, 20):
        ab = np.array([origin_a, origin_b])
        while err >= eps and iters < mxite:
            iters += 1
            p = three_para_logisical(theta, ab[0], ab[1], gamma=gamma)

            # ai偏导
            fa = D / (1 - c) * np.sum((theta - ab[1]) * (p - c) * (y - p) / p)

            # bi偏导
            fb = -D * ab[0] / (1 - c) * np.sum((y - p) * (p - c) / p)

            # ai bi导
            fab = -D / (1 - c) * np.sum(
                (p - c) * ((y / p - 1) + D * ab[0] / (1 - c) * (theta - ab[1]) * (1 - p) / p * (y * c / p - p)))

            # ai二阶导
            ffa = (D / (1 - c)) ** 2 * np.sum((theta - ab[1]) ** 2 * (p - c) * (1 - p) / p * (y * c / p - p))

            # bi二阶导
            ffb = (D * ab[0] / (1 - c)) ** 2 * np.sum((p - c) * (1 - p) / p * (y * c / p - p))

            hessian = np.mat(([ffa, fab], [fab, ffb]))  # 2 * 2
            gradient = np.mat([fa, fb])  # 1 * 2
            dia = np.identity(np.shape(hessian)[0]) * lbd
            hessian = hessian - dia
            inverse = hessian.I

            step = np.array(np.dot(inverse, gradient.T).T).reshape([2, ])
            step = step_reduce_ab(theta, y, ab, step, gamma=gamma, low_bound=low_bound, up_bound=up_bound)
            err = abs(step.mean())
            ab = ab - step

            ab[ab < low_bound] = low_bound[ab < low_bound]
            ab[ab > up_bound] = up_bound[ab > up_bound]

        new_lkhd = likehood(theta, ab[0], ab[1], y, gamma=gamma)
        if new_lkhd > lkhd:
            lkhd = new_lkhd
            result_ab = ab

    return err, iters, result_ab[0], result_ab[1]


def step_reduce_ab(theta, y, ab0, step, gamma=1, low_bound=np.array([-3.0, -3.0]), up_bound=np.array([3.0, 3.0])):
    '''
    下山法
    :param theta:
    :param y:
    :param ab0:
    :param step:
    :param gamma:
    :return:
    '''
    like0 = likehood(theta, ab0[0], ab0[1], y, gamma=gamma)
    ab = ab0 - step
    ab[ab < low_bound] = low_bound[ab < low_bound]
    ab[ab > up_bound] = up_bound[ab > up_bound]
    like1 = likehood(theta, ab[0], ab[1], y, gamma=gamma)
    while True:
        if like1 < like0:
            step = step / 2.0
            # print 'like1', like1
            # print 'like0', like0
            # print '减半', step
            ab = ab0 - step
            ab[ab < low_bound] = low_bound[ab < low_bound]
            ab[ab > up_bound] = up_bound[ab > up_bound]
            like1 = likehood(theta, ab[0], ab[1], y, gamma=gamma)
        else:
            break
    return step
