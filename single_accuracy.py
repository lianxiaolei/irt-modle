#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from newton import *

reload(sys)
sys.setdefaultencoding('utf8')


def get_accuracy(origin_file, lost_file, result_file):
    origin = np.loadtxt(origin_file)
    lost = np.loadtxt(lost_file)
    result = np.loadtxt(result_file)
    total_err = 0
    lost_err = 0
    lostnum = 0
    numbers = origin.shape[0] * origin.shape[1]
    for i in range(np.shape(origin)[0]):
        for j in range(np.shape(origin)[1]):
            if lost[i, j] == 2.0:
                lostnum += 1
                lost_err += abs(origin[i, j] - result[i, j])
                continue
            total_err += abs(lost[i, j] - result[i, j])
    print '误差个数', total_err
    print '缺失个数', lostnum
    print '正确率', (numbers - lostnum - total_err) / (numbers - lostnum)
    print '测试集正确率', (lostnum - lost_err) / lostnum


def make_param(c=0.1, gamma=1):
    n = 100
    m = 200
    u = np.zeros((m, n))
    theta = np.random.normal(0, 1, n).reshape([1, n])
    a = np.random.lognormal(0, 0.3, m).reshape([m, 1])
    b = np.random.normal(0, 1, m).reshape([m, 1])
    # np.savetxt('theta.txt', theta)
    # np.savetxt('a.txt', a)
    # np.savetxt('b.txt', b)
    p = c + (gamma - c) / (1 + np.exp(-1.7 * a * (theta - b)))
    print np.shape(p)

    rand = np.random.random(np.shape(p))
    print np.shape(rand)

    u[rand <= p] = 1
    u[rand > p] = 0
    np.savetxt('nanswer.txt', u, fmt='%i')
    for i in range(5000):
        u[(int(np.random.random() * 200), int(np.random.random() * 100))] = 2
    np.savetxt('nanswer_lost.txt', u, fmt='%i')


if __name__ == '__main__':
    get_accuracy('nanswer.txt', 'nanswer_lost.txt', 're_answer.txt')
    # make_param(c=0.25, gamma=1)
