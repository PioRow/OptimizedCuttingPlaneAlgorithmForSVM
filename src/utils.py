import numpy as np
from scipy.optimize import minimize

def reduced_problem_solver(alphas, betas, C, M):

    x= np.zeros(M+1)
    def obj(x):
        return 0.5 * np.dot(x[:M], x[:M]) + C * x[M]
    def create_constrains(alphas, betas, M):
        con=[
            {
                "type":"ineq",
                "fun":lambda x: x[M]-np.dot(x[:M], a)-b
            }
            for a,b in zip(alphas.T, betas)
        ]
        con.append({
            "type": "ineq",
            "fun": lambda x: x[M]
        })
        return con

    constr= create_constrains(alphas, betas, M)

    res=minimize(obj,x,constraints=constr,method="SLSQP")
    if res.success:
        return res.x[:M], res.x[M]
    else:
        raise ValueError("Optimization failed: " + res.message)