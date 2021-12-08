# -*-coding:utf-8 -*-
# 目标求解2*sin(x)+cos(x)最大值
import random
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# 超参数取值范围，二的次方
import nn_classifier

super_parameter_bin_len = nn_classifier.super_parameter_bin_len


def to_decimal(m_chromosome):
    m_id = 0
    temp = []
    for bin_len in super_parameter_bin_len:
        one_para = 0
        for _ in range(bin_len):
            one_para *= 2
            one_para += m_chromosome[m_id]
            m_id += 1
        temp.append(one_para)
    return temp


def to_binary(super_para_list):
    m_chromosome = []
    for i, para in enumerate(super_para_list, 0):
        # 先变成对应长度的二进制字符串
        bs = ("{:0>%db}" % super_parameter_bin_len[i]).format(para)
        print('整数', para, '二进制', bs)
        for c in bs:
            m_chromosome.append(1 if c == '1' else 0)
    return m_chromosome


class GA(object):
    # 初始化种群 生成chromosome_length大小的population_size个个体的种群

    def __init__(self, generation, pop_size, chrome_len, cross_rate, mutate_rate):

        self.generation = generation
        self.pop_size = pop_size
        self.chromosome_len = chrome_len
        # self.population=[[]]
        self.cross_rate = cross_rate  # 交叉概率
        self.mutate_rate = mutate_rate  # 变异概率
        # self.fitness_value=[]

    # 从二进制到十进制
    # 编码  input:种群,染色体长度 编码过程就是将多元函数转化成一元函数的过程
    def translation(self, population):

        value_list = []
        for m_chromosome in population:
            value_list.append(to_decimal(m_chromosome))
        # 一个染色体编码完成，由一个二进制数编码为一个十进制数
        return value_list

    # 返回种群中所有个体编码完成后的十进制数

    # from protein to function,according to its functoin value

    # a protein realize its function according its structure
    # 目标函数相当于环境 对染色体进行筛选，这里是2*sin(x)+math.cos(x)
    def get_fitness(self, m_generation, population, best_fitness):
        fitness_list = []
        para_list_of_population = self.translation(population)
        for i, para_list in enumerate(para_list_of_population, 1):
            if i == 1 and best_fitness > -1:
                fitness = best_fitness
                print("上一代最佳个体，直接使用最优值%f " % best_fitness)
            else:
                print('\n计算第 %d 代 %d 个个体的适应度' % (m_generation, i))
                fitness = nn_classifier.run(para_list)
            print('第 %d 代 第%d个个体，适应度为%f' % (m_generation, i, fitness))
            fitness_list.append(fitness)

        # 这里将sin(x)作为目标函数
        return np.array(fitness_list)

    # 计算适应度和

    def sum(self, fitness_value):
        total = 0

        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total

    # 计算适应度斐伯纳且列表
    def cumsum(self, fitness1):
        for i in range(len(fitness1) - 2, -1, -1):
            # range(start,stop,[step])
            # 倒计数
            total = 0
            j = 0

            while (j <= i):
                total += fitness1[j]
                j += 1

            fitness1[i] = total
            fitness1[len(fitness1) - 1] = 1

    def crossover(self, p1, pop):
        if np.random.rand() < self.cross_rate:
            # 随机选取一个进行交叉, 已经进行过选择过，是放回抽样，所以适应度大的在列表中会有多个副本，交叉概率大
            p2 = pop[np.random.randint(0, len(pop))]
            log = False
            id = int(np.random.rand() * len(p1))
            if (np.random.rand() < 0.0001):
                log = True
                print("\n打印交叉情况")
                print(p1, p2, sep='\n')
                print(id)
            for i, x in enumerate(p1, id):
                if i >= len(p1):
                    break
                p1[i], p2[i] = p2[i], p1[i]
            if log:
                print(p1, p2, sep='\n')

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

    # 寻找最好的适应度和个体

    def best(self, population, fitness_value):
        bestindividual = population[0]
        bestfitness = fitness_value[0]
        # print(fitness_value)

        for i in range(1, len(population)):
            # 循环找出最大的适应度，适应度最大的也就是最好的个体
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                bestindividual = population[i]

        return bestindividual, bestfitness

    def evolve(self):
        # population = self.species_origin()
        pop = np.vstack([np.random.randint(0, 2, size=sum(super_parameter_bin_len)) for _ in range(self.pop_size)])
        one_epoch_visual = []
        best_individual_value, in_best_fitness = None, -1
        last_fitness = -1
        no_opt_count = 0
        for i in range(generation):
            fitness_value = self.get_fitness(i + 1, pop, in_best_fitness)
            # print('fit funtion_value:',function_value)
            # print('fitness_value:',fitness_value)

            best_individual, in_best_fitness = self.best(pop, fitness_value)
            best_individual_value = to_decimal(best_individual)
            one_epoch_visual.append(in_best_fitness)
            # if i % 10 == 0:
            print('第%d代, 最佳个体为：' % (i + 1), nn_classifier.get_real_para(best_individual_value))
            print('适应度为%f' % in_best_fitness)
            # 如果目标函数的值多次没有增加，那么终止演化
            if in_best_fitness - last_fitness <= 0:
                no_opt_count += 1
            else:
                no_opt_count = 0
            last_fitness = in_best_fitness
            if no_opt_count > generation / 3:
                print('第%d代，收敛几乎停止，终止演化' % i)
                return best_individual_value, in_best_fitness, one_epoch_visual

            idx = self.select(pop, fitness_value)
            pop = pop[idx]
            pop_copy = pop.copy()
            for parent in pop:
                self.crossover(parent, pop_copy)
                self.mutate(parent)
            pop[0] = best_individual
        return best_individual_value, in_best_fitness, one_epoch_visual


if __name__ == '__main__':
    population_size = 4
    generation = 10
    chromosome_length = torch.sum(torch.tensor(super_parameter_bin_len))
    pc = 0.6
    pm = 0.1
    ga = GA(generation, population_size, chromosome_length, pc, pm)
    total_visual_data = []
    best_individual_list = []
    best_fitness_list = []
    for _ in range(1):
        print('第%d轮演化:{}' % (_ + 1))
        best_individual, best_fitness, one_visual_data = ga.evolve()
        plt.plot(one_visual_data)
        total_visual_data.append(one_visual_data)
        best_individual_list.append(best_individual)
        best_fitness_list.append(best_fitness)
        print("\n这一轮最佳个体为", best_individual)
        print("适应度为", best_fitness)
        print('\n\n')
    final_best_fitness = max(best_fitness_list)
    max_id = best_fitness_list.index(final_best_fitness)
    print('最终最佳个体为', best_individual_list[max_id])
    print('适应度为：%d' % final_best_fitness)
    plt.show()
    plt.plot(best_fitness_list)
    plt.show()
    plt.plot(best_individual_list)
    plt.show()
