import functions
from enumFunctions import Functions
from GWO import GWO
import pandas as pd
def gwo():
    # f=open("run.txt","w")
    # list of objective functions: no damavandi function!!! ;;;; todo
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.rotatedhyperellipsoid),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    bounds=[[-32.768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-30,30],[-5,10],[-2048,2048],[0,14]]
    #obj_func = objective_functions[0]
    #obj_func = functions.selectFunction(Functions.schwefel)
    # dim array size, -5 lb +5 lb 
    #GWO(obj_func, -500, 500, 30, 250, 250,3)
    dfs = [pd.DataFrame()] * 9
    # pop_sizes=[25,50,100,500,1000]
    # num_of_gens=[25,50,100,500,1000]
    pop_sizes=[25]
    num_of_gens=[25]
    a_decrease=[4,3,2]
    results=[]
    for index,bound in enumerate(bounds):
        obj_func=objective_functions[index]
        for pop_size in pop_sizes:
            for num_gen in num_of_gens:
                for a in a_decrease:
                    for _ in range(5):
                        # print(obj_func,bound[0],bound[1],5,pop_size,num_gen,a)
                        best=GWO(obj_func,bound[0],bound[1],5,pop_size,num_gen,a)
                        results.append([obj_func.__name__,pop_size,num_gen,a,best[1]])
                    df=pd.DataFrame(results,columns=['obj_f','pop_size','num_gen','a_decreaese','best_fitness'])
                    df['std_dev']=df['best_fitness'].std()
                    df['avg_fitness']=df['best_fitness'].mean()
                    df['max_fitness']=df['best_fitness'].min()
                    print(df)
                    dfs[index]=df
                results=[]
    # f.close()
    with pd.ExcelWriter('output.xlsx') as writer:
        for idx,df in enumerate(dfs):
            dfs[idx].to_excel(writer, sheet_name=objective_functions[idx].__name__)
def main():
    gwo()

if __name__ == "__main__":
    main()
    
#def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):