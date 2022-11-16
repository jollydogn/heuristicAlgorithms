# hill climbing search of the ackley objective function
import numpy as np
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import functions
from enumFunctions import Functions
import pandas as pd

# objective function
# def objective(v):
# 	x, y = v
# 	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

def objective( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point
	solution = None
	while solution is None or not in_bounds(solution, bounds):
		solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	for i in range(n_iterations):
		# take a step
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

def bounds(function_num,dimension):
    lower_bounds=[None]*dimension
    upper_bounds=[None]*dimension
    bounds=[[-32.768,32768],[-600,600],[-500,500],[-5.12,5.12],[-5.12,5.12],[-65.536,65.536],[-30,30],[-5,10],[-2048,2048],[0,14]]
    boundsFinal=[None]*dimension
    for idx in range(dimension):
        boundsFinal[idx]=bounds[function_num]
        
        
    return asarray(boundsFinal)
def allocateObjectiveFunctions():
    objective_functions=[functions.selectFunction(Functions.ackley),functions.selectFunction(Functions.griewank),functions.selectFunction(Functions.schwefel),functions.selectFunction(Functions.rastrigin),functions.selectFunction(Functions.sphere),functions.selectFunction(Functions.rotatedhyperellipsoid),functions.selectFunction(Functions.perm),functions.selectFunction(Functions.zakharov),functions.selectFunction(Functions.rosenbrock),functions.selectFunction(Functions.damavandi)]
    return objective_functions

def main():
        # seed the pseudorandom number generator
        seed(5)
        # define range for input
        n_iterationss=[250, 500, 1000, 1500, 2000, 5000, 10000]
        obj_functions =allocateObjectiveFunctions()
        o_functions=obj_functions
        results=[]
        dfs = [pd.DataFrame()] * 9
        
# =============================================================================
#         n_iterations = [250, 500, 1000, 1500, 2000, 5000, 10000]        # define the total iterations
# =============================================================================
        # define the maximum step size
        step_size = 0.05
        for idx,obj_f in enumerate(o_functions):
            boundFor=bounds(idx,30)
            for hmcr_idx,iteration in enumerate(n_iterationss):
                for _ in range(5):
                        best, score = hillclimbing(obj_f, boundFor, iteration, step_size)
                        print('Done!')
                        print('f(%s) = %f' % (best, score))
                        results.append([obj_f.__name__,iteration,boundFor[0][0],boundFor[0][1],score])
                df=pd.DataFrame(results,columns=['obj_f','numberOfGeneration','lowerBound','upperBound','score'])
                df['std_dev'] = df['score'].rolling(5).std()
                df['avg_fitness']=df['score'].rolling(5).mean()
                df['max_fitness']=df['score'].rolling(5).min()
                print(df)
                dfs[idx]=df
            results=[]
        with pd.ExcelWriter('output.xlsx') as writer:
            for idx,df in enumerate(dfs):
                dfs[idx].to_excel(writer, sheet_name=obj_functions[idx].__name__)
        # perform the hill climbing search
        
        
main()