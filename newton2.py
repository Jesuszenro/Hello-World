import numpy as np


def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

############## Example 1

print("\n Example 1 \n")

f = lambda x: x**2 - 2
f_prime = lambda x: 2*x

estimate = newton(f, f_prime, 1.5, 1e-6, 10)
print("estimate =", estimate)
print("sqrt(2) =", np.sqrt(2))

############## Example 2

print("\n Example 2 \n")

f = lambda x: x**3 + 4*x**2 -10
f_prime = lambda x: 3*x**2 + 8*x

estimate = newton(f, f_prime, 1.5, 1e-4, 10)
print("estimate =", estimate)


############## Example 3

print("\n Example 3 \n")

f = lambda x: np.sqrt(x)-np.cos(x)
f_prime = lambda x: (1/2)*x**(-1/2)+np.sin(x)

estimate = newton(f, f_prime, 0.5, 1e-3, 10)
print("estimate =", estimate)




