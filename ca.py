#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

if __name__ == '__main__':
    answer = np.loadtxt('answer.txt')
    # for que in answer:
    #     print que

    a = np.array([2, 3]).reshape([2, 1])
    b = np.array([4, 5]).reshape([1, 2])
    c = np.array([10, 12]).reshape([1, 2])
    print a
    print b
    print a - b
