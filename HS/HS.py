#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 9:50
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : HS.py
# @Statement : Harmony Search
# @Reference : A new heuristic optimization algorithm: Harmony search[J]. Simulation, 2001, 2(2):60-68
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, tan ,cosh, tanh, sinh, abs, exp, mean, pi, prod, sqrt, sum
import functions
from enumFunctions import Functions
import pandas as pd
# def obj(x):
#     """
#     The objective function of pressure vessel design
#     :param x:
#     :return:
#     """
#     x1 = x[0]
#     x2 = x[1]
#     x3 = x[2]
#     x4 = x[3]
#     g1 = -x1 + 0.0193 * x3
#     g2 = -x2 + 0.00954 * x3
#     g3 = -math.pi * x3 ** 2 - 4 * math.pi * x3 ** 3 / 3 + 1296000
#     g4 = x4 - 240
#     if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
#         return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
#     else:
#         return 1e10

def obj( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)


def boundary_check(value, lb, ub):
    """
    The boundary check
    :param value:
    :param lb: the lower bound (list)
    :param ub: the upper bound (list)
    :return:
    """
    for i in range(len(value)):
        value[i] = max(value[i], lb[i])
        value[i] = min(value[i], ub[i])
    return value


def harmonySearch(hms, iter, hmcr, par, bw, nnew, lb, ub):
    """
    The main function of the HS
    :param hms: harmony memory size
    :param iter: the number of iterations
    :param hmcr: harmony memory consideration rate
    :param par: pitch adjustment rate
    :param bw: bandwidth
    :param nnew: the number of new harmonies created in each iteration
    :param lb: the lower bound (list)
    :param ub: the upper bound (list)
    :return:
    """
    # Step 1. Initialization
    pos = []  # the set of harmonies
    score = []  # the score of harmonies
    dim = len(lb)  # dimension
    for i in range(hms):
        temp_pos = [random.uniform(lb[j], ub[j]) for j in range(dim)]
        pos.append(temp_pos)
        score.append(obj(temp_pos))
    iter_best = []
    gbest = min(score)  # the score of the best-so-far harmony
    gbest_pos = pos[score.index(gbest)].copy()  # the best-so-far harmony
    con_iter = 0

    # Step 2. The main loop
    for t in range(iter):
        new_pos = []
        new_score = []

        # Step 2.1. Create new harmonies
        for _ in range(nnew):
            temp_pos = []
            for j in range(dim):
                if random.random() < hmcr:  # utilize harmony memory
                    ind = random.randint(0, hms - 1)
                    temp_pos.append(pos[ind][j])
                    if random.random() < par:  # pitch adjustment
                        temp_pos[j] += random.normalvariate(0, 1) * bw * (ub[j] - lb[j])
                else:
                    temp_pos.append(random.uniform(lb[j], ub[j]))
            temp_pos = boundary_check(temp_pos, lb, ub)
            new_pos.append(temp_pos)
            new_score.append(obj(temp_pos))

        # Step 2.2. Update harmony memory
        new_pos.extend(pos)
        new_score.extend(score)
        sorted_score = sorted(new_score)
        pos = []
        score = []
        for i in range(hms):
            score.append(sorted_score[i])
            pos.append(new_pos[new_score.index(sorted_score[i])])

        # Step 2.3. Update the global best
        if score[0] < gbest:
            gbest = score[0]
            gbest_pos = pos[0].copy()
            con_iter = t + 1
        iter_best.append(gbest)

    # Step 3. Sort the results
    x = [i for i in range(iter)]
    plt.figure()
    plt.plot(x, iter_best, linewidth=2, color='blue')
    plt.xlabel('Iteration number')
    plt.ylabel('Global optimal value')
    plt.title('Convergence curve')
    plt.show()
    return {'best score': gbest, 'best solution': gbest_pos, 'convergence iteration': con_iter}

def allocateObjectiveFunctions():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    return objective_functions

def bounds(function_num,dimension):
    lower_bounds=[None]*dimension
    upper_bounds=[None]*dimension
    bounds=[[-32768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-30,30],[-5,10],[-2048,2048],[0,14]]
    
    for idx in range(dimension):
        lower_bounds[idx]=bounds[function_num][0]
        upper_bounds[idx]=bounds[function_num][1]
    return lower_bounds, upper_bounds
    
def optimize():
    hms = [250,500,1000,5000,10000]
    bw=[0.1, 0.15, 0.2, 0.25, 0.30]
    hmcr=[0.90,0.92, 0.95, 0.97, 0.99]
    par=[0.15,0.2,0.25,0.3,0.35,0.4]
    iter=1000
    nnew=20
    lb,ub=bounds(1,30)

    dfs = [pd.DataFrame()] * 9
    print(harmonySearch(hms[0], iter, hmcr[0], par[0], bw[0], nnew, lb, ub))
    # def harmonySearch(hms, iter, hmcr, par, bw, nnew, lb, ub):


def main():
    optimize()
    
main()
## params
# if __name__ == '__main__':
#     # Parameter settings
#     # hms = 30
#     # iter = 1000
#     # hmcr = 0.9
#     # par = 0.1
#     # bw = 0.02
#     # nnew = 20
#     # #-32768,32768
#     # lb = [-32768,-32768,-32768,-32768]
#     # ub = [32768,32768,32768,32768]
#     # print(main(hms, iter, hmcr, par, bw, nnew, lb, ub))
    
#     lower,upper=bounds(2,5)
#     print(lower,upper)
# 
# 
#     