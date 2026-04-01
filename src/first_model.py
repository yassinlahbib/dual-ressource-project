import gurobipy as gp
from gurobipy import GRB
from utils import read_file, gantt_chart
import numpy as np
import time


class Instance:
    def __init__(self, nb_jobs, max_nb_operations, max_nb_sub_operations, nb_workers, levels_workers, difficulty_jobs, difficulty_operations, processing_time_operations, constraints_precedence_operations, constraints_precedence_sub_operations):
        self.nb_jobs = nb_jobs # int
        self.max_nb_operations = max_nb_operations # int
        self.max_nb_sub_operations = max_nb_sub_operations # int
        self.nb_workers = nb_workers # int
        self.levels_workers = levels_workers # len = nb_workers
        self.difficulty_jobs = difficulty_jobs # len = nb_jobs
        self.difficulty_operations = difficulty_operations # size(nb_jobs, max_nb_operations)
        self.processing_time_operations = processing_time_operations # size(nb_jobs, max_nb_operations)
        self.constraints_precedence_operations = constraints_precedence_operations # size(nb_jobs, max_nb_operations, max_nb_operations)
        self.constraints_precedence_sub_operations = constraints_precedence_sub_operations # size(nb_jobs, max_nb_operations, max_nb_sub_operations, max_nb_sub_operations)

    def __str__(self):
        res = (f"Number of jobs: {self.nb_jobs}\n"
               f"Max number of operations: {self.max_nb_operations}\n"
               f"Max number of sub-operations: {self.max_nb_sub_operations}\n" 
               f"Number of workers: {self.nb_workers}\n"
               f"Worker levels: len={len(self.levels_workers)}\n {self.levels_workers}\n"
               f"Job difficulties: len= {len(self.difficulty_jobs)}\n{self.difficulty_jobs}\n"
               f"Operation difficulties: shape= {self.difficulty_operations.shape}\n{self.difficulty_operations}\n"
               f"Operation processing times: shape= {self.processing_time_operations.shape}\n{self.processing_time_operations}\n"
               f"Precedence constraints: shape= {self.constraints_precedence_operations.shape}\n{self.constraints_precedence_operations}\n"
               f"Sub-operation precedence constraints: shape= {self.constraints_precedence_sub_operations.shape}\n{self.constraints_precedence_sub_operations}\n")
        return res


class Model:
    def __init__(self, instance):
        self.instance = instance

    def _build_model(self):
        m = gp.Model("dual_ressource_scheduling")

        ##### Variables
        x = m.addVars(self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.max_nb_sub_operations, self.instance.nb_workers, vtype=GRB.BINARY, name="x") # x[i, j, s, k] = 1 if sub-operation s of operation j of job i is assigned to worker k
        d = m.addVars(self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.max_nb_sub_operations, self.instance.nb_workers, vtype=GRB.INTEGER, name="d") # d[i, j, s, k] = starting time of sub-operation s of operation j of job i if assigned to worker k
        C = m.addVars(self.instance.nb_jobs, vtype=GRB.INTEGER, name="C") # C[i] = completion time of job i if minimize sum of C[i], completion time of all jobs if minimize C_max
        C_max = m.addVar(vtype=GRB.INTEGER, name="C_max") # C_max = makespan
        delta = m.addVars(self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.max_nb_sub_operations, self.instance.max_nb_sub_operations, self.instance.nb_workers, vtype=GRB.BINARY, name="delta") # delta[i, j, h, g, s, z, k] = 1
        # f = m.addVars(self.instance.nb_jobs, self.instance.max_nb_operations, vtype=GRB.INTEGER, name="f") # f[i,j] = completion time of operation j of job i if assigned to worker k -- Pour etre sur que la fin de l'opération est avant le début de l'opération suivante dans les contraintes de précédence



        ##### Constraints 

        # Au plus 1 worker par sous operation pour le moment, mettre a 2 plus tard
        # constraint : sum_k x[i, j, s, k] <= 1 for all i, j, s !!!!!!!!!
        # max one worker per operation
        # constraint (8) but just 1 max per operation
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    m.addConstr((gp.quicksum(x[i, j, s, k] for k in range(self.instance.nb_workers)) <= 1), name=f"max_sub_operation_assignment_{i}_{j}_{s}")


        # constraint : if processing time of operation O[i,j,s] is 0 then x[i, j, s, k] = 0 for all k
        # to evit assigning workers to non existing operations
        # constraint (11)
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    print(i,j,s, self.instance.processing_time_operations[i][j][s])
                if self.instance.processing_time_operations[i][j][s] == 0:
                    print("operation", (i,j,s) ,"n'existe pas " )
                    for k in range(self.instance.nb_workers):
                        m.addConstr((x[i,j,s,k] == 0), name=f"zero_processing_time_{i}_{j}_{s}_{k}")


        # constraint : sum_k x[i, j, s, k] >= 1 for all i, j, s
        # at least one worker per sub-operation (for existing sub-operations)
        # constraint (7)
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    if self.instance.processing_time_operations[i][j][s] > 0:
                        m.addConstr((gp.quicksum(x[i, j, s, k] for k in range(self.instance.nb_workers)) >= 1), name=f"min_sub_operation_assignment_{i}_{j}_{s}")

        # Hypothèse !!!!!!!!! : Seulement un opérateur est assigné à l'opération O[i,j] -> sinon faudrait faire d[i,j,k] + y[i,j,k], avec y[i,j,k] temps max si 2 opérateurs sur l'opération O[i,j]
        # constraint : C[i] >= d[i, j, k] + processing_time_operations[i, j] for all i, j, k
        # Completion time of each job
        # constraint (3)
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    for k in range(self.instance.nb_workers):
                        print(f"C[{i}] >= d[{i}, {j}, {s}, {k}] + processing_time_operations[{i}, {j}, {s}] = {self.instance.processing_time_operations[i][j][s]}")
                        m.addConstr((C[i] >= d[i, j, s, k] + self.instance.processing_time_operations[i, j, s]), name=f"completion_time_{i}_{j}_{s}_{k}")


        # constraint : C_max >= C[i] for all i
        # constraint (4)
        for i in range(self.instance.nb_jobs):
            m.addConstr((C_max >= C[i]), name=f"makespan_{i}")


        # constraint (overlap) : d[i,j,s,k] >= f[h,g,,k] - M * delta[i,j,h,g,s,z,k] for all i, j, h, g, s, z, k with (i,j,s) != (h,g,z)
        #                        d[h,g,z,k] >= f[i,j,s,k] - M * (1 - delta[i,j,h,g,s,z,k]) for all i, j, h, g, s, z, k with (i,j,s) != (h,g,z)
        # constraint (5) (6)
        M = 10000
        for k in range(self.instance.nb_workers):
            for i in range(self.instance.nb_jobs):
                for j in range(self.instance.max_nb_operations):
                    for h in range(self.instance.nb_jobs):
                        for g in range(self.instance.max_nb_operations):
                            for s in range(self.instance.max_nb_sub_operations):
                                for z in range(self.instance.max_nb_sub_operations):
                                    if (i, j, s) != (h, g, z):
                                        f_hgzk = d[h,g,z,k] + self.instance.processing_time_operations[h, g, z] * x[h, g, z, k] 
                                        f_ijsk = d[i,j,s,k] + self.instance.processing_time_operations[i, j, s] * x[i, j, s, k]
                                        m.addConstr((d[i,j,s,k] >= f_hgzk - M * delta[i,j,h,g,s,z,k]), name=f"overlap1_{i}_{j}_{h}_{g}_{s}_{z}_{k}")
                                        m.addConstr((d[h,g,z,k] >= f_ijsk - M * (1 - delta[i,j,h,g,s,z,k])), name=f"overlap2_{i}_{j}_{h}_{g}_{s}_{z}_{k}")


        # constraint : if x[i, j, k] = 0 then d[i, j, k] = 0 for all i, j, k
        # contrainte big M pour forcer d[i,j,k] à 0 si x[i,j,k] = 0
        # constraint (10)
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    for k in range(self.instance.nb_workers):
                        print(f"d[{i}, {j}, {s}, {k}] <= M * x[{i}, {j}, {s}, {k}]")
                        m.addConstr((d[i,j,s,k] <= M * x[i,j,s,k]), name=f"start_time_zero_if_not_assigned_{i}_{j}_{s}_{k}")



        # constraint : precedence constraints of operations for the same job
        # f_ijk = d_ijk + processing_time_operation_ij * x_ijk
        # f_ijk <= sum_(k' in W) (d_ij'k')
        # constraint (12)
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for j_prime in range(self.instance.max_nb_operations):
                    for s in range(self.instance.max_nb_sub_operations):
                        for k in range(self.instance.nb_workers):
                            if (j != j_prime) and (self.instance.constraints_precedence_operations[i,j,j_prime] == 1): # if O_ij must be performed before operation O_ij'
                                f_ijsk = d[i,j,s,k] + self.instance.processing_time_operations[i,j,s] * x[i,j,s,k] # actif pour 2 workers au maximum
                                # contrainte juste à la ligne suivante marche si une seul personne est affecté à la sous opération O(ij's) si deux personnes la somme n'a plus d'effet
                                # m.addConstr((f_ijsk <= gp.quicksum(d[i,j_prime,s,k] for k in range(self.instance.nb_workers))), name=f"precedence_operations_{i}_{j}_{j_prime}_{s}_{k}")
                                for k_prime in range(self.instance.nb_workers):
                                    for s_prime in range(self.instance.max_nb_sub_operations):
                                        m.addConstr((f_ijsk <= M * (1 - x[i,j_prime,s_prime,k_prime]) + d[i,j_prime,s_prime,k_prime] ), name=f"precedence_operations_inactive_{i}_{j}_{j_prime}_{s}_{k}")  # contrainte pour désactiver la contrainte de précédence si x[i,j,s,k] = 0



                       
        # constraint : precedence constraints of sub-operations for the same operation
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    for s_prime in range(self.instance.max_nb_sub_operations):
                        for k in range(self.instance.nb_workers):
                            if (s != s_prime) and (self.instance.constraints_precedence_sub_operations[i,j,s,s_prime] == 1): # if sub-operation s of operation O_ij must be performed before sub-operation s' of operation O_ij
                                f_ijsk = d[i,j,s,k] + self.instance.processing_time_operations[i,j,s] * x[i,j,s,k]
                                # m.addConstr((f_ijsk <= gp.quicksum(d[i,j,s_prime,k] for k in range(self.instance.nb_workers))), name=f"precedence_sub_operations_{i}_{j}_{s}_{s_prime}_{k}")
                                for k_prime in range(self.instance.nb_workers):
                                    # for s_prime in range(self.instance.max_nb_sub_operations):
                                    m.addConstr((f_ijsk <= M * (1 - x[i,j,s_prime,k_prime]) + d[i,j,s_prime,k_prime] ), name=f"precedence_sub_operations_inactive_{i}_{j}_{s}_{s_prime}_{k}")  # contrainte pour désactiver la contrainte de précédence si x[i,j,s,k] = 0
        


        # constraint : level of worker k must be >= difficulty of the sub operation assigned to worker k
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    for k in range(self.instance.nb_workers):
                        m.addConstr((x[i,j,s,k] * self.instance.levels_workers[k] >= self.instance.difficulty_operations[i,j,s] * x[i,j,s,k]), name=f"worker_with_capacity{i}_{j}_{s}_{k}")

        # constraint : if two worker k1 and k2 are assigned to the same sub-operation (i,j,s) then the starting time of the sub-operation for both workers must be the same : d[i,j,s,k1] = d[i,j,s,k2]
        for i in range(self.instance.nb_jobs):
            for j in range(self.instance.max_nb_operations):
                for s in range(self.instance.max_nb_sub_operations):
                    for k1 in range(self.instance.nb_workers):
                        for k2 in range(k1+1, self.instance.nb_workers): # k2 > k1 to avoid duplicate constraints
                            print(f"d[{i}, {j}, {s}, {k1}] <= d[{i}, {j}, {s}, {k2}] + M * (2 - x[{i}, {j}, {s}, {k1}] + x[{i}, {j}, {s}, {k2}])")
                            print(f"d[{i}, {j}, {s}, {k2}] <= d[{i}, {j}, {s}, {k1}] + M * (2 - x[{i}, {j}, {s}, {k1}] + x[{i}, {j}, {s}, {k2}])")
                            m.addConstr((d[i,j,s,k1] <= d[i,j,s,k2] + M * (2 - (x[i,j,s,k1] + x[i,j,s,k2]))), name=f"same_start_time1_{i}_{j}_{s}_{k1}_{k2}")
                            m.addConstr((d[i,j,s,k2] <= d[i,j,s,k1] + M * (2 - (x[i,j,s,k1] + x[i,j,s,k2]))), name=f"same_start_time2_{i}_{j}_{s}_{k1}_{k2}")

        
        m.setObjective(C_max, GRB.MINIMIZE)

        # # minimsier somme des complétudes des jobs
        # m.setObjective(gp.quicksum(C[i] for i in range(self.instance.nb_jobs)), GRB.MINIMIZE)
        
        print("------------------")
        print("------------------")
        print("------------------")
        m.write("../results/model.lp")
        print("------------------")
        print("------------------")
        print("------------------")
        # time.sleep(2)
        

        return m

    def solve(self):
        m = self._build_model()
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found with objective value:", m.objVal)
            m.write("../results/solution.sol")
        else:
            print("No optimal solution found. Status code:", m.status)
            return
        

        ##### build solution object
        all_vars = m.getVars()
        values = m.getAttr('X', all_vars)
        names = m.getAttr('VarName', all_vars)
        res = []
        for name, value in zip(names, values):
            res.append((name, value))
        
        print("objective value:", m.objVal)
        # time.sleep(2)
        return Solution(res, self.instance)

class Solution:
    def __init__(self, var_list, instance):
        self.x = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.nb_workers)) # x[i, j, s, k] = 1 if sub operation s of  operation j of job i is assigned to worker k, 0 otherwise
        self.d = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.nb_workers))
        self.C = np.zeros(instance.nb_jobs)
        self.C_max = 0
        self.delta = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.max_nb_sub_operations, instance.nb_workers))
        
        print("var_list", var_list)
        for v in var_list:

            if v[0][0][0] == "x":
                indices = v[0][2:-1].split(",") # x[i, j, s, k] -> indices = [i, j, s, k]
                i, j, s, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
                self.x[i, j, s, k] = v[1]
                print(f"x[{i}, {j}, {s}, {k}] = {v[1]}")


            elif v[0][0] == "d" and v[0][1] == "[" : # == "[" pour éviter confusion avec variable delta
                indices = v[0][2:-1].split(",") # d[i, j, s, k] -> indices = [i, j, s, k]
                i, j, s, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
                self.d[i, j, s, k] = v[1]


            elif v[0][0] == "C" and v[0][1] != "_": # C[i] -> indices = [i]
                indices = v[0][2:-1].split(",")
                i = int(indices[0])
                self.C[i] = v[1]


            elif v[0] == "C_max":
                self.C_max = v[1]
            

            elif v[0][0] == "d" and v[0][1] == "e" : # delta[i, j, h, g, s, z, k] -> indices = [i, j, h, g, s, z, k]
                indices = v[0][6:-1].split(",")
                i, j, h, g, s, z, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3]), int(indices[4]), int(indices[5]), int(indices[6])
                self.delta[i, j, h, g, s, z, k] = v[1]

    def __str__(self):
        res = (f"x: {self.x.shape} \n{self.x}\n"
               f"d: {self.d.shape} \n{self.d}\n"
               f"C: {self.C.shape} \n{self.C}\n"
               f"C_max: {self.C_max}\n"
               f"delta: {self.delta.shape}\n{self.delta}\n")
        return res


if __name__ == "__main__":
    nb_jobs, max_nb_operations, max_nb_sub_operations, nb_workers, levels_workers, difficulty_jobs, difficulty_operations, processing_time_operations, constraints_precedence_operations, constraints_precedence_sub_operations = read_file("../data/data_2.test")
    instance = Instance(nb_jobs, max_nb_operations, max_nb_sub_operations, nb_workers, levels_workers, difficulty_jobs, difficulty_operations, processing_time_operations, constraints_precedence_operations, constraints_precedence_sub_operations)

    print(instance)
    model = Model(instance)
    s = model.solve()
    print(s)


    gantt_chart(s, instance, color_print="sub_operation")
    