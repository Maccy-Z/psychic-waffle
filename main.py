from enum import Flag, auto

class P_Types(Flag):
    NORMAL = auto() # Standard PDE: u = f(us, x) enforced on this point.
    BOUNDARY = auto()   # Dirichlet: u = Dirichlet(x) enforced on this point.
    DERIV = auto()  # deriv(u, us) = Neumann(x) enforced on this point. Can be used as central Neumann derivative or edge, depending on other nodes.
    GHOST = auto() # No function on point. Ghost point. u function inherited from central DERIV point.

    BOTH = BOUNDARY | DERIV
    GRAD = NORMAL | GHOST # u requires fitting on point.


print(P_Types.NORMAL in P_Types.GRAD)
print(P_Types.BOTH in P_Types.DERIV)

