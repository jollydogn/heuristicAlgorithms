import functions
from enumFunctions import Functions
import GA
import pandas as pd
import numpy as np

#test function
def GeneticAlgorithm():
    functionIndex = Functions.schwefel 
    _lb = -500
    _ub = 500
    dim = 6
    pop_size = 250 # population size
    maxiter = 250
    obj_func = functions.selectFunction(Functions.schwefel)
    sol = GA.GA(obj_func, _lb, _ub, dim, pop_size, maxiter,0.0001,0,1)
    return sol

def allocateObjectiveFunctions():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.rotatedhyperellipsoid),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    return objective_functions

def bounds(function_num):
    bounds=[[-32.768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-30,30],[-5,10],[-2048,2048],[0,14]]
    return bounds[function_num][0], bounds[function_num][1]

def writeDfToXlsxFile(dfs, obj_functions):
    with pd.ExcelWriter('output.xlsx') as writer:
        for idx,df in enumerate(dfs):
            dfs[idx].to_excel(writer, sheet_name=obj_functions[idx].__name__)


def optimizeGA():
    dimension=30
    pop_sizes=[250,500,1000,3000]
    num_of_generations=[250, 500, 1000, 1500]
    mut_probs=np.array([0.01, 0.02, 0.05, 0.1, 0.15])
    #============================
    #   crossover types:
    #   0: 1-point crossover
    #   1: 2-point crossover
    #   2: uniform crossover
    #============================
    crossover_types=[0,1,2]
    # create a dictionary for output
    crossover_dict={
        "0": "1-point_crossover",
        "1": "2-point_crossover",
        "2": "uniform_crossover"
    }
    #============================
    #   selection types:
    #   0:roulette wheel
    #   1:tournament selection
    #============================
    selection_types=[0,1]
    # create a dictionary for output
    selection_dict={
        "0": "roulette_wheel",
        "1": "tournament_selection"
    }
    obj_functions = allocateObjectiveFunctions()
    results=[]
    dfs=[pd.DataFrame()]*len(obj_functions)
    for idx,obj_f in enumerate(obj_functions):
        lower_bound,upper_bound=bounds(idx)
        for pop_size in pop_sizes:
            for num_of_gen in num_of_generations:
                for mut_prob in mut_probs:
                    for crossover_type in crossover_types:
                        for selection_type in selection_types:
                            for _ in range(5):
                                sol = GA.GA(obj_f, lower_bound, upper_bound, dimension, pop_size, num_of_gen,mut_prob,crossover_type,selection_type)
                                best=sol.best
                                results.append([obj_f.__name__, lower_bound, upper_bound, dimension, pop_size, num_of_gen,mut_prob,crossover_dict[str(crossover_type)],selection_dict[str(selection_type)],best])
        df=pd.DataFrame(results,columns=['obj_f','lower_bound','upper_bound','dimension','pop_size','num_of_gen','mut_prob','crossover_type','selection_type','best'])
        df['std_dev'] = df['best'].rolling(5).std()
        df['avg_fitness']=df['best'].rolling(5).mean()
        df['max_fitness']=df['best'].rolling(5).min()
        df.drop('best', axis=1, inplace=True)
        dfs[idx]=df.iloc[4::5]
        results=[]
    writeDfToXlsxFile(dfs,obj_functions)

def main():
    optimizeGA()

if __name__ == "__main__":
    main()