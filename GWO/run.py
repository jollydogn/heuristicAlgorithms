import functions
from enumFunctions import Functions
from GWO import GWO
import pandas as pd
from numba import jit, cuda

@jit(target ="cuda")
def allocateObjectiveFunctions():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.rotatedhyperellipsoid),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    return objective_functions
@jit(target ="cuda")
def bounds(function_num):
    bounds=[[-32.768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-30,30],[-5,10],[-2048,2048],[0,14]]
    return bounds[function_num][0], bounds[function_num][1]
@jit(target ="cuda")
def optimizeGWO():
    dim=30
    obj_functions=allocateObjectiveFunctions()
    pop_sizes=[250,500,1000,3000]
    num_of_gens=[250, 500, 1000, 1500]
    a_decrease=[4,3,2]
    dfs = [pd.DataFrame()] * 10
    results=[]
    for idx,_ in enumerate(obj_functions):
        obj_func=obj_functions[idx]
        lower_bound,upper_bound=bounds(idx)
        for pop_size in pop_sizes:
            for num_gen in num_of_gens:
                for a in a_decrease:
                    for _ in range(5):
                        best=GWO(obj_func,lower_bound,upper_bound,dim,pop_size,num_gen,a)
                        results.append([obj_func.__name__,lower_bound,upper_bound,pop_size,num_gen,a,best[1]])
        df=pd.DataFrame(results,columns=['obj_f','lower_bound','upper_bound','pop_size','num_gen','a_decreaese','best_fitness'])
        df['std_dev']=df['best_fitness'].rolling(5).std()
        df['avg_fitness']=df['best_fitness'].rolling(5).mean()
        df['max_fitness']=df['best_fitness'].rolling(5).min()
        df.drop('best_fitness', axis=1, inplace=True)
        dfs[idx]=df.iloc[4::5]
        results=[]

    with pd.ExcelWriter('output.xlsx') as writer:
        for idx,df in enumerate(dfs):
            dfs[idx].to_excel(writer, sheet_name=obj_functions[idx].__name__)

@jit(target ="cuda")
def main():
    optimizeGWO()

if __name__ == "__main__":
    main()
    
#def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):