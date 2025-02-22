from gpkit import Variable, VectorVariable, Model
from gpkit.nomials import Monomial, Posynomial, PosynomialInequality

x = Variable("x")
y = Variable("y")
z = Variable("z")
w = Variable("w", )

S = 200
objective = 1/(x*y*z)  + x ** 2 + y ** 3
constraints = [2*x*y + 2*x*z + 2*y*z <= S,
               x >= 2*y]
m = Model(objective, constraints)

sol = m.solve(verbosity=5)
print(sol.table())
