#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from newton import *

reload(sys)
sys.setdefaultencoding('utf8')

c = 0.25
D = 1.7


def draw_theta(a, b, label, gamma=1):
    plt.figure()
    theta = np.arange(-3, 3, 0.001)

    lk = [likehood(t, a, b, label, gamma=gamma) for t in theta]
    plt.plot(theta, lk)
    print '极大值theta,likehood', theta[np.array(lk).argmax()], np.max(lk)
    plt.show()


def draw_ab(theta, b, label, gamma):
    plt.figure()
    a = np.arange(-3, 3, 0.001)
    lk = [likehood(theta, i, b, label, gamma=gamma) for i in a]
    plt.plot(a, lk)
    print '极大值a,b,likehood', a[np.array(lk).argmax()], b, np.max(lk)
    plt.show()


def run(answer, student, question_a, question_b, gamma=1, minerr=0.0001, eps=1e-4, newton_mxite=30, lbd=1e-4,
        ab_low=np.array([0.0, -3.0]), ab_up=np.array([2.0, 3.0]), theta_low=-3.0, theta_up=3.0, max_ite=10):
    totallk = -np.inf  # 总似然初始值
    tmperr = 1.0
    ite_num = 0  # 循环次数初始值

    while tmperr > minerr and ite_num < max_ite:
        ite_num += 1
        lk = []  # ab更新后总似然

        for j in range(np.shape(answer)[0]):
            stu = answer[j, :]
            theta = []  # 学生能力向量
            y = []  # 答题标签
            stu = list(stu)

            for index in range(len(stu)):
                if not stu[index] == 2.0:
                    theta.append(student[index])
                    y.append(stu[index])

            theta = np.array(theta).reshape((len(theta), 1))
            y = np.array(y).reshape((len(y), 1))

            err, iters, a, b = \
                newton_for_ab(theta, y, origin_a=question_a[j], origin_b=question_b[j],
                              gamma=gamma, eps=eps, mxite=newton_mxite, lbd=lbd, low_bound=ab_low,
                              up_bound=ab_up)
            question_a[j] = a
            question_b[j] = b

            like = likehood(theta, a, b, y, gamma=gamma)
            lk.append(like)

        tmperr = abs(sum(lk) - totallk)
        totallk = sum(lk)

        lk = []  # 更新theta后总似然
        for i in range(np.shape(answer)[1]):
            que = answer[:, i]
            y = []
            tmpa = []
            tmpb = []
            que = list(que)

            for index in range(len(que)):
                if not que[index] == 2:
                    y.append(que[index])
                    tmpa.append(question_a[index])
                    tmpb.append(question_b[index])

            tmpa = np.array(tmpa).reshape(len(tmpa), 1)
            tmpb = np.array(tmpb).reshape(len(tmpb), 1)
            y = np.array(y).reshape((len(y), 1))

            err, iters, theta = \
                newton_for_theta(tmpa, tmpb, y, origin_t=student[i], gamma=gamma, mxite=newton_mxite, low_bound=theta_low,
                                 up_bound=theta_up)
            student[i] = theta
            like = likehood(theta, tmpa, tmpb, y, gamma=gamma)
            lk.append(like)

        tmperr = abs(sum(lk) - totallk)
        totallk = sum(lk)

    return question_a, question_b, student, totallk


def boot(answer, init_a=1.0, init_b=0.0, init_theta=1.0, gamma=1, minerr=0.001, eps=1e-4, newton_mxite=30, lbd=1e-4,
                                        ab_low=np.array([0.0, -3.0]), ab_up=np.array([2.0, 3.0]),
                                        theta_low=-3.0, theta_up=3.0, max_ite=50):
    '''
    跑算法
    :param answer:
    :param init_a:
    :param init_b:
    :param init_theta:
    :param gamma:
    :param minerr:
    :param eps:
    :param newton_mxite:
    :param lbd:
    :param ab_low:
    :param ab_up:
    :param theta_low:
    :param theta_up:
    :param max_ite:
    :return:
    '''
    student = []
    question_a = []
    question_b = []
    '''学生能力向量初始化'''

    for i in range(answer.shape[1]):
        student.append(init_theta)
    for i in range(answer.shape[0]):
        question_a.append(init_a)
        question_b.append(init_b)

    vec_a, vec_b, vec_theta, like = run(answer, student, question_a, question_b,
                                        gamma=gamma, minerr=minerr, eps=eps, newton_mxite=newton_mxite, lbd=lbd,
                                        ab_low=ab_low, ab_up=ab_up,
                                        theta_low=theta_low, theta_up=theta_up, max_ite=max_ite)

    return vec_a, vec_b, vec_theta, like


def get_guess(a, b, theta):
    '''
    使用参数预测答题结果
    :param a:
    :param b:
    :param theta:
    :return:
    '''
    tmplist0 = []
    for i in range(len(vec_a)):
        tmplist1 = []
        for j in range(len(vec_theta)):
            p = three_para_logisical(theta=theta[j], a=a[i], b=b[i], gamma=1)
            print p
            tmplist1.append(int(round(p, 0)))
        tmplist0.append(tmplist1)
    tmparr = np.array(tmplist0)
    # np.savetxt('re_answer.txt', tmparr, fmt='%i')
    return tmparr


if __name__ == '__main__':

    '''加载答题矩阵'''
    answer = np.loadtxt('nanswer_lost.txt')
    vec_a, vec_b, vec_theta, like = boot(answer=answer, init_a=1.0, init_b=0.0, init_theta=1.0, gamma=1, minerr=0.001, eps=1e-4, newton_mxite=30, lbd=1e-4,
                                        ab_low=np.array([0.0, -3.0]), ab_up=np.array([2.0, 3.0]),
                                        theta_low=-3.0, theta_up=3.0, max_ite=50)
    forecasting = get_guess(vec_a, vec_b, vec_theta)
    print vec_a
    print vec_b
    print vec_theta
    print like
