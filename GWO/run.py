import functions
from enumFunctions import Functions
from GWO import GWO
def gwo():
    f=open("run.txt","w")
    # list of objective functions: no damavandi function!!! ;;;; todo
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.ellipse),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock)]
    #obj_func = objective_functions[0]
    #obj_func = functions.selectFunction(Functions.schwefel)
    # dim array size, -5 lb +5 lb 
    #GWO(obj_func, -500, 500, 30, 250, 250,3)
    bounds=[[-32768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-5,5],[-5,10],[-2048,2048]]
    pop_sizes=[250,500,1000,5000,10000]
    num_of_gens=[250,500,1000,5000,10000]
    a_decrease=[4,3,2]
    for index,bound in enumerate(bounds):
        obj_func=objective_functions[index]
        print("index:",index)
        for pop_size in pop_sizes:
            for num_gen in num_of_gens:
                for a in a_decrease:
                    print(obj_func,bound[0],bound[1],5,pop_size,num_gen,a)
                    GWO(obj_func,bound[0],bound[1],5,pop_size,num_gen,a)
    f.close()
def main():
    gwo()

if __name__ == "__main__":
    main()
    
#def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):