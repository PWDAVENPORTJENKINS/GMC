"""
P. W. Davenport-Jenkins
University of Manchester
MSc Econometrics
2019-07-08
"""
import warnings
import scipy.optimize as optimize
from functools import partial
import numpy as np
import math
import time
from scipy.optimize import LinearConstraint, Bounds
import operator


def moment_conditions(beta, x, y, z):
    return (np.add(y, -beta * x) * z).T

"""

The optimisation is over a vector, P say, where P[0] = beta, the model parameter
and P[:1] is the vector of sample probabilities which are initially set to all be 1/T.

"""
def GMC_constraint_one(x):
    # Constrain the sum of all elements, except the first, to be equal to 1
    # That is the sum of the sample probabilities is equal to one 
    return np.sum(x[1:]) - 1


def GMC_constraint_two(moments, x):
    # Here we want that the products of the i^th sample probability
    # and the i^th moment vectors all sum to zero.  
    m = x[1:]
    return sum(moments * m[:, np.newaxis])


def GMC_objective_function(T, x):
    return np.mean(-np.log(T * x[1:]))


def GMC(beta, x, y, z, K, T):
    mu = np.ones(T) / T
    initial_params = np.insert(mu, 0, beta)
    moments_matrix = moment_conditions(beta, x, y, z)
    lower_bound = np.ones(T) * 1e-3
    lower_bound = np.insert(lower_bound, 0, -np.inf)
    upper_bound = np.ones(T)
    upper_bound = np.insert(upper_bound, 0, np.inf)
    feasible_bounds = Bounds(lower_bound, upper_bound, keep_feasible=True)
    constraint_two = partial(GMC_constraint_two, moments_matrix)

    constraint = [
                    {'type': 'eq', 'fun': GMC_constraint_one},
                    {'type': 'eq', 'fun': constraint_two}
                 ]

    cost = partial(GMC_objective_function, T)
    result = optimize.minimize(
        cost,
        initial_params,
        constraints=constraint,
        bounds=feasible_bounds,
        method='SLSQP', # trust-constraint doesnt seem to work
        options={'maxiter': 1000, 'disp': True}
    )
    return result.x

T = 100
K = 5
R_SQUARED = 0.01
RHO = 0.25
BETA = 0

eta = math.sqrt(R_SQUARED / (K * (1 - R_SQUARED)))
pi = eta * np.ones(K)
error_mean = np.array([0, 0])
error_varcov = np.array([[1, RHO], [RHO, 1]])

z_mean = np.zeros(K)
z_varcov = np.identity(K)
z = np.random.multivariate_normal(z_mean, z_varcov, T).T
epsilon, nu = np.random.multivariate_normal(error_mean, error_varcov, T).T

x = z.T @ pi + nu
y = BETA * x + epsilon

start_time = time.time()

beta_guess = 0.5
values_GMC = GMC(beta_guess, x, y, z, K, T)

print("--- Total Time: {} seconds ---".format(time.time() - start_time))
