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
    sol = GA.GA(obj_func, _lb, _ub, dim, pop_size, maxiter)
    return sol

def main():
    sol = GeneticAlgorithm()

if __name__ == "__main__":
    main()