import functions
from enumFunctions import Functions
import numpy as np
from SA import simulated_annealing_linear
from SA import simulated_annealing_geometric
from openpyxl import Workbook, load_workbook
import array
import pandas as pd
import time
import statistics as stat

def SimulatedAnnealingLinear():
    obj_func = functions.selectFunction(Functions.ackley)
    # dim array size, -5 lb +5 lb 
    simulated_annealing_linear( min_values = [-32768,-32768,-32768,-32768,-32768], max_values = [32768,32768,32768,32768,32768], mu = 0, sigma = 1, initial_temperature = 100.0, temperature_iterations = 100,
        final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)
    
def SimulatedAnnealingGeometric():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    bounds=[[-32768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-30,30],[-5,10],[-2048,2048],[0,14]]
    temperatures = [1000,5000,10000]
    results = []
    dfs = [pd.DataFrame()] * 9
    for index,bound in enumerate(bounds):
        obj_func=objective_functions[index]
        for temp in temperatures:
            lower = np.empty(30)
            lower.fill(bound[0])
            
            upper = np.empty(30)
            upper.fill(bound[1])
            
            for _ in range(5):
                best = simulated_annealing_geometric( min_values = lower, max_values = upper, mu = 0, sigma = 1, initial_temperature = temp, temperature_iterations = 100,
                final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)
                results.append([obj_func.__name__,temp,best[0][29]])
            #my_array = np.array([[obj_func,temp,best[0][29]]])
            #df1 = pd.DataFrame(my_array, columns = ['obj_f','temp','geometric'])
            df=pd.DataFrame(results,columns = ['obj_f','temp','geometric'])
            #df['std_dev'] = df.apply(lambda _: '', axis=1)
           # df.iloc[-1]['std_dev']=df['geometric'].std()
            # df.iloc[-1:]['std_dev']=df['geometric'].std()
            
            df['std_dev']=df['geometric'].std()
            df['avg_fitness']=df['geometric'].mean()
            df['max_fitness']=df['geometric'].max()
            dfs[index] = df

        results=[]
    with pd.ExcelWriter('output.xlsx') as writer:
        for idx,df in enumerate(dfs):  
            dfs[idx].to_excel(writer, sheet_name=objective_functions[idx].__name__)

            #df.append(df2,ignore_index=True)
            # # dim array size, -5 lb +5 lb 
            # simulated_annealing_geometric( min_values = lower, max_values = upper, mu = 0, sigma = 1, initial_temperature = temp, temperature_iterations = 100,
            #     final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)
            
            # SimulatedAnnealingGeometric(obj_func,bound[0],bound[1],30,temp)
            
            
    
    

def main():
    # temperatures = [1000,5000,10000]
    # for i in temperatures:
    #     SimulatedAnnealingGeometric(i)
    SimulatedAnnealingGeometric()

if __name__ == "__main__":
    main()