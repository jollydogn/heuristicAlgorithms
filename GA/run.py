import functions
from enumFunctions import Functions
import GA

def GeneticAlgorithm():
    functionIndex = Functions.schwefel 
    _lb = -500
    _ub = 500
    dim = 6
    pop_size = 250 # population size
    maxiter = 250
    obj_func = functions.selectFunction(Functions.schwefel)
    sol = GA.GA(obj_func, _lb, _ub, dim, pop_size, maxiter,0.0001,1,0)
    return sol

# def GeneticAlgorithm():
#     objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
#     bounds=[[-32768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-30,30],[-5,10],[-2048,2048],[0,14]]
#     pop_sizes=[250,500,1000,5000,10000]
#     num_of_generations=[250,500,1000,1500,2000]
#     mut_probabilities=[0.01,0.02,0.05,0.1,0.15]
    

def main():
    sol = GeneticAlgorithm()

if __name__ == "__main__":
    main()