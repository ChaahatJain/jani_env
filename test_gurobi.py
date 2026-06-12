import os

print("PWD:", os.getcwd())
print("GRB_LICENSE_FILE:", os.environ.get("GRB_LICENSE_FILE"))
print("License file exists:", os.path.exists(os.environ.get("GRB_LICENSE_FILE", "")))

try:
    import gurobipy as gp
    print("gurobipy version:", gp.gurobi.version())

    m = gp.Model()
    x = m.addVar(lb=0, name="x")
    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.addConstr(x <= 1)
    m.optimize()

    print("Status:", m.Status)
    print("Objective:", m.ObjVal)
    print("GUROBI_WORKS")
except Exception as e:
    print("GUROBI_FAILED")
    print(type(e).__name__, e)
    raise