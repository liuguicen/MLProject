# -*-coding:utf-8 -*-
# 目标求解2*sin(x)+cos(x)最大值
import random
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# 超参数取值范围，二的次方
from matplotlib.font_manager import FontProperties

import nn_classifier

# 参数的二进制位长度，代表了求解的精度
PARA_LEN = 16
NORMAL_SIZE = 2 ** (PARA_LEN - 1)


def decode(m_chromosome):
    """

               解码，将二进制染色体，变成神经网络的参数，先变成整数，再归一化
           """
    start = 0
    para_list = []
    while start < len(m_chromosome):
        one_para = 0.
        for i in range(PARA_LEN):
            one_para *= 2
            one_para += m_chromosome[start + i]
        start += PARA_LEN
        para_list.append((one_para - NORMAL_SIZE) / NORMAL_SIZE)  # 归一化到-2-2之间
    return para_list


class GA(object):
    def __init__(self, generation, pop_size, cross_rate, mutate_rate):
        self.generation = generation
        self.pop_size = pop_size
        self.cross_rate = cross_rate  # 交叉概率
        self.mutate_rate = mutate_rate  # 变异概率

    def get_fitness(self, m_generation, population):
        # 二进制解码变成十进制的参数列表
        para_list = [decode(m_chromosome) for m_chromosome in population]
        fitness_list = []
        accuracy_list = []
        for i, para_list in enumerate(para_list, 1):
            # print('\n计算第 %d 代 %d 个适应度\n ' % (m_generation, i))
            fitness = nn_classifier.run(para_list)
            accuracy = nn_classifier.get_test_accuracy()
            # print('第 %d 代 第%d个个体，适应度为%f准确率为%.2f' % (m_generation, i, fitness, accuracy))
            fitness_list.append(fitness)
            accuracy_list.append(accuracy)
        return np.array(fitness_list), np.array(accuracy_list)

    def crossover(self, p1, pop):
        """
                  单点交叉
                  p1 交叉的父辈1
               """
        if np.random.rand() < self.cross_rate:
            # 随机选取一个进行交叉, 前面的选择是放回抽样，所以适应度大的在列表中会有多个副本，发生交叉的概率大，符合适者生存
            p2 = pop[np.random.randint(0, len(pop))]
            log = False
            id = int(np.random.rand() * len(p1))
            for i, x in enumerate(p1, id):
                if i >= len(p1):
                    break
                p1[i], p2[i] = p2[i], p1[i]

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.rand() < self.mutate_rate:
                child[i] = 0 if child[i] == 1 else 1

    def select(self, pop, fitness):
        """
            选择种群
            Args:
                fitness: 种群适应度
            Returns:
                idx: 选中个体在种群中的下标
            使用的是轮盘赌选择算法
            在生成的0-pop_size数组中，每个元素以fitness / fitness.sum()放回抽样，抽pop_size个
        """
        fit_sum = sum(fitness)
        idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True, p=fitness / fit_sum)
        return idx

    def evolve(self):
        # population = self.species_origin()
        pop = np.vstack(
            [np.random.randint(0, 2, size=nn_classifier.get_para_size() * PARA_LEN) for _ in range(self.pop_size)])
        best_fitness_list = []
        best_loss_list = []
        best_accuracy_list = []
        for i in range(generation):
            fitness_list, accuracy_list = self.get_fitness(i + 1, pop)
            argmax = fitness_list.argmax()
            best_individual, best_fitness, best_accuracy = pop[argmax], fitness_list[argmax], accuracy_list[argmax]

            best_fitness_list.append(best_fitness)
            best_loss_list.append(1. / best_fitness - 1)
            best_accuracy_list.append(best_accuracy)
            print('第%d代, 最佳个体预测准确率为%.2f\n 适应度为%f' % (i + 1, best_accuracy, best_fitness))
            idx = self.select(pop, fitness_list)
            pop = pop[idx]
            pop_copy = pop.copy()
            for parent in pop:
                self.crossover(parent, pop_copy)
                self.mutate(parent)
            pop[0] = best_individual
        return best_fitness_list, best_loss_list, best_accuracy_list


def visual_result(epoch_loss_list, best_fitness_list, best_accuracy_list):
    """
    画出结果
    """
    plt.xlabel('generation')
    plt.ylabel('fitness')
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    plt.title(u'每一轮演化的适应度', fontproperties=font)
    plt.show()

    plt.xlabel('generation')
    plt.ylabel('loss')
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    plt.title(u'每一轮演化的损失值', fontproperties=font)
    for one_loss in epoch_loss_list:
        plt.plot(one_loss)
    plt.show()

    plt.xlabel('epoch')
    plt.ylabel('best fitness')
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    plt.title(u'每一轮最优适应度', fontproperties=font)
    plt.plot(best_fitness_list)
    plt.show()

    plt.xlabel('epoch')
    plt.ylabel('best accuracy')
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    plt.title(u'每一轮最高准确率', fontproperties=font)
    plt.plot(best_accuracy_list)
    plt.show()


if __name__ == '__main__':
    epoch = 10
    population_size = 100
    generation = 200
    pc = 0.6
    pm = 0.1
    ga = GA(generation, population_size, pc, pm)
    best_fitness_list = []
    best_accuracy_list = []
    epoch_loss_list = []
    for _ in range(10):
        print('第%d轮演化:{}' % (_ + 1))
        fitness_list, loss_list, accuracy_list = ga.evolve()
        plt.plot(fitness_list)
        epoch_loss_list.append(loss_list)
        best_fitness_list.append(max(fitness_list))
        ba = max(accuracy_list)
        best_accuracy_list.append(ba)
        print("\n这一轮最佳个体准确度为%.2f\n\n" % ba)
    final_best_accuracy = max(best_accuracy_list)
    print('最终最佳准确率为：%.2f' % final_best_accuracy)
    visual_result(epoch_loss_list, best_fitness_list, best_accuracy_list)
