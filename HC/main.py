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

def main():
        # seed the pseudorandom number generator
        seed(5)
        # define range for input
        bounds = asarray([[-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768],
                          [-32768, 32768]
                          ])
        # define the total iterations
        n_iterations = 250
        # define the maximum step size
        step_size = 0.05
        # perform the hill climbing search
        best, score = hillclimbing(objective, bounds, n_iterations, step_size)
        print('Done!')
        print('f(%s) = %f' % (best, score))
        
main()