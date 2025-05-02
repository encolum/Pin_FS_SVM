import cplex
print(cplex.__version__)
cpx = cplex.Cplex()
print(cpx.get_license_type())