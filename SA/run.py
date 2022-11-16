import functions
from enumFunctions import Functions
import numpy as np
from SA import simulated_annealing_linear
from SA import simulated_annealing_geometric
import pandas as pd
    
def allocateObjectiveFunctions():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.rotatedhyperellipsoid),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    return objective_functions

def bounds(function_num,dimension):
    lower_bounds=[None]*dimension
    upper_bounds=[None]*dimension
    bounds=[[-32.768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-30,30],[-5,10],[-2048,2048],[0,14]]
    
    for idx in range(dimension):
        lower_bounds[idx]=bounds[function_num][0]
        upper_bounds[idx]=bounds[function_num][1]
    return lower_bounds, upper_bounds

def SimulatedAnnealing():
    dim=30
    obj_functions=allocateObjectiveFunctions()
    temperatures = [1000,5000,10000]
    results = []
    dfs = [pd.DataFrame()] * 10
    for idx,_ in enumerate(obj_functions):
        obj_func=obj_functions[idx]
        lower,upper=bounds(idx,30)
        for temp in temperatures:
            modes=[0,1]
            for mode in modes:
                for i in range(5):
                    if(mode==0):
                        best = simulated_annealing_geometric( min_values = lower, max_values = upper, mu = 0, sigma = 1, initial_temperature = temp, temperature_iterations = 5000,
                        final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)
                        results.append([obj_func.__name__,lower[0],upper[0],temp,round(best[0, -1], 4),'geometric'])

                    else:
                        best = simulated_annealing_linear( min_values = lower, max_values = upper, mu = 0, sigma = 1, initial_temperature = temp, temperature_iterations = 5000,
                        final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)
                        results.append([obj_func.__name__,lower[0],upper[0],temp,round(best[0, -1], 4),'linear'])
                        
                
        df=pd.DataFrame(results,columns = ['obj_f','lower_bound','upper_bound','temp','best','type'])
        df['std_dev'] = df['best'].rolling(5).std()
        df['avg_fitness']=df['best'].rolling(5).mean()
        df['max_fitness']=df['best'].rolling(5).min()
        df.drop('best', axis=1, inplace=True)
        dfs[idx]=df.iloc[4::5]
        results=[]

    with pd.ExcelWriter('output.xlsx') as writer:
        for idx,df in enumerate(dfs):  
            dfs[idx].to_excel(writer, sheet_name=obj_functions[idx].__name__)
                        
def main():
    SimulatedAnnealing()
    
    

if __name__ == "__main__":
    main()