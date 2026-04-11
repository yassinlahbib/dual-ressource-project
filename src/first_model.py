import gurobipy as gp
from gurobipy import GRB
from utils import read_file, gantt_chart, plot_levels_workers
import numpy as np
import time

class Instance:
    def __init__(self, res):

        # INT
        self.nb_jobs = res["nb_jobs"] # int
        self.nb_professions = res["nb_professions"] # int
        self.nb_sub_operations_profession = res["nb_sub_operations_profession"] # int
        self.nb_sub_operations = res["nb_sub_operations"] # int
        self.max_nb_operations = res["max_nb_operations"] # int
        self.max_nb_sub_operations = res["max_nb_sub_operations"] # int
        self.nb_workers = res["nb_workers"] # int

        # DIFFICULTIES AND TIMES
        self.sub_operations_difficulties = res["sub_operations_difficulties"] # size (nb_sub_operations)
        self.sub_operations_times = res["sub_operations_times"] # size (nb_sub_operations, 3)

        # LEVELS OF WORKERS
        self.levels_workers = res["levels_workers"] # size (nb_workers, nb_professions)
        
        # JOBS STRUCTURE
        self.jobs_struct = res["jobs_struct"] # len=nb_jobs, len(jobs_struct[i]) = number of operations of job i, len(jobs_struct[i][j]) = number of sub-operations of operation j of job i, jobs_struct[i][j][s] = index of sub-operation s of operation j of job i
        self.difficulty_jobs = res["difficulty_jobs"] # size (nb_jobs)

        # CONSTRAINTS
        self.constraints_precedence_operations = res["constraints_precedence_operations"] # size(nb_jobs, max_nb_operations, max_nb_operations)
        self.constraints_precedence_sub_operations = res["constraints_precedence_sub_operations"] # size(nb_jobs, max_nb_operations, max_nb_sub_operations, max_nb_sub_operations)
        
        # MAPPING SUB-OP TO METIER
        self.sub_op_to_m = res["dict_sub_op_to_m"]

    def __str__(self):

        jobs_struct_str = ""
        for i in range(self.nb_jobs):
            jobs_struct_str += f"Job {i} : \n"
            for j in range(len(self.jobs_struct[i])):
                jobs_struct_str += f"\tO_{j} : "
                for s in range(len(self.jobs_struct[i][j])):
                    jobs_struct_str += f"{self.jobs_struct[i][j][s]} "
                jobs_struct_str += "\n"

        res = (f"-------------------------------------\n"
               f"Start of Instance:\n"
               f"-------------------------------------\n"
               f"Number of jobs: {self.nb_jobs}\n"
               f"Number of professions: {self.nb_professions}\n"
               f"Number of sub-operations per profession: {self.nb_sub_operations_profession}\n"
               f"Max number of operations per Jobs: {self.max_nb_operations}\n"
               f"Max number of sub-operations per operation: {self.max_nb_sub_operations}\n" 
               f"Total number of sub-operations: {self.nb_sub_operations}\n"
               f"Total number of workers: {self.nb_workers}\n"
               f"Sub-operation to profession mapping:\n{self.sub_op_to_m}\n"


               f"Worker levels: shape={self.levels_workers.shape}\n {self.levels_workers}\n"
               f"Job difficulties: shape= {self.difficulty_jobs.shape}\n{self.difficulty_jobs}\n"
               f"Sub-operation difficulties: shape= {self.sub_operations_difficulties.shape}\n{self.sub_operations_difficulties}\n"
               f"Sub-operation processing times: shape= {self.sub_operations_times.shape}\n{self.sub_operations_times}\n")
        res += (f"Jobs structure: len= {len(self.jobs_struct)}\n{jobs_struct_str}\n"
               f"Precedence constraints: shape= {self.constraints_precedence_operations.shape}\n{self.constraints_precedence_operations}\n"
               f"Sub-operation precedence constraints: shape= {self.constraints_precedence_sub_operations.shape}\n{self.constraints_precedence_sub_operations}\n"
               f"-------------------------------------\n"
               f"End of Instance:\n"
               f"-------------------------------------\n")
        return res


class Model:
    def __init__(self, instance):
        self.instance = instance

    def _build_model(self, objective, weight, priority):
        """
        Construit le modèle de programmation linéaire
        
        Args:
            objective (int): l'objectif à optimiser, peut être "makespan", "skill", "both" ou "lexicographic"
            weight (list): les poids à accorder à chaque objectifs (makespan, skill) si objective = "both", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (makespan, skill) si objective = "lexicographic", n'est pas utilisé sinon
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            
        Returns:
            m (gp.Model): le modèle de programmation linéaire construit
        """
        m = gp.Model(f"dual_resource_scheduling_objective_{objective}")

        #######################################################################
        ######################## FEASABILITY VARIABLES ########################
        #######################################################################

        # Useful for use uniquely the existing tuples of indices for the variables
        # Allow to reduce the number of variables and constraints
        keys = [] # for varibles x, d, f, Delta_min
        keys_without_k = [] # for variable Level_min
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    keys_without_k.append((i, j, s))
                    for k in range(self.instance.nb_workers):
                         keys.append((i, j, s, k))

        keys_delta = []
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for h in range(self.instance.nb_jobs):
                    for g in range(len(self.instance.jobs_struct[h])):
                        for s in range(len(self.instance.jobs_struct[i][j])):
                            for z in range(len(self.instance.jobs_struct[h][g])):
                                for k in range(self.instance.nb_workers):
                                    keys_delta.append((i, j, h, g, s, z, k))

        keys_z = []
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    keys_z.append((i, j, s, 0)) # 1 si O_ijs est fait en solo
                    keys_z.append((i, j, s, 1)) # 2 si O_ijs est fait en apprentissage
                    keys_z.append((i, j, s, 2)) # 3 si O_ijs est fait en collab


        # Count number of sub_operations to do
        nb_tasks = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    nb_tasks += 1


        #######################################################################
        ############################## VARIBALES ##############################
        #######################################################################

        x = m.addVars(keys, vtype=GRB.BINARY, name="x") # x[i, j, s, k] = 1 if sub-operation s of operation j of job i is assigned to worker k
        d = m.addVars(keys, vtype=GRB.CONTINUOUS, name="d") # d[i, j, s, k] = starting time of sub-operation s of operation j of job i if assigned to worker k
        C = m.addVars(self.instance.nb_jobs, vtype=GRB.CONTINUOUS, name="C") # C[i] = completion time of job i if minimize sum of C[i], completion time of all jobs if minimize C_max
        C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max") # C_max = makespan
        delta = m.addVars(keys_delta, vtype=GRB.BINARY, name="delta") # delta[i, j, h, g, s, z, k] = 1
        # y = m.addVars(self.instance.nb_workers,self.instance.nb_sub_operations, vtype=GRB.INTEGER, name="y") # y[k] = number of sub-operations assigned to worker k
        l = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="l") # l[k,m] = level of worker k before performing metier m after run of the PL
        f = m.addVars(keys, vtype=GRB.CONTINUOUS, name="f") # f[i,j,s,k] = completion time of operation j of job i if assigned to worker k -- Ajout de cette variable pour prendre en compte le fait que la duré d'une tache peut etre different selon si fait en solo, en apprentissage ou en collab
        # nb_w_to = m.addVars(keys_without_k, vtype=GRB.INTEGER, name="nb_w_to_ijs") # nb_w_to_ijs[i,j,s] = number of workers assigned to O_ijs
        z_auxilary = m.addVars(keys_z, vtype=GRB.INTEGER, name="z_auxilary") # z[i,j,s,0] = 1 if O_ijs is done in solo, z[i,j,s,1] = 1 if O_ijs is done in apprentissage
        
        # Linearisation min pour savoir si une tache est fait en apprentissage ou en collab
        Level_min = m.addVars(keys_without_k, vtype=GRB.CONTINUOUS, name="Level_min") #vaut le level min d'un worker sur O_ijs
        Delta_min = m.addVars(keys, vtype=GRB.BINARY, name="Delta_min") # pour linearisation du min


        M = 10000
        #######################################################################
        ############################## CONSTRAINTS ############################
        #######################################################################

        

        # constraint : sum_k x[i, j, s, k] <= 2 for all i, j, s 
        # Hypothèse : first sub-operation of first operation of each job must be performed by only one worker
        # At most 2 workers per sub-op. For the fistr sub-op of each job at most 1 worker for we know it is the worker assigned to the all job
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])): # for all operations except the first one of each job
                for s in range(len(self.instance.jobs_struct[i][j])):
                    m.addConstr((gp.quicksum(x[i, j, s, k] for k in range(self.instance.nb_workers)) <= 2), name=f"max_sub_operation_assignment_{i}_{j}_{s}")
        
        for i in range(self.instance.nb_jobs):
            m.addConstr((gp.quicksum(x[i, 0, 0, k] for k in range(self.instance.nb_workers)) <= 1), name=f"max_first_sub_operation_assignment_{i}")


        ## No need constraints below now, because we use only the existing tuples of indices for the variables x and d
        # # constraint : if processing time of operation O[i,j,s] is 0 then x[i, j, s, k] = 0 for all k
        # # --> new beacause new structure -> 0 to inexisiting 
        # # to evit assigning workers to non existing operations
        # # constraint (11)
        # for i in range(self.instance.nb_jobs):
        #     for j in range(self.instance.max_nb_operations):
        #         for s in range(self.instance.nb_sub_operations):
        #             if self.instance.jobs_struct[i][j][s] == 0: # if the sub operation does not exist
        #                 print("operation", (i,j,s) ,"n'existe pas " )
        #                 for k in range(self.instance.nb_workers):
        #                     m.addConstr((x[i,j,s,k] == 0), name=f"non_existing_sub_operation_{i}_{j}_{s}_{k}")
  

        # constraint : sum_k x[i, j, s, k] >= 1 for all i, j, s
        # Hypothesise: all sub-op are assigned !!!!!!!!!!!!!
        # at least one worker per sub-operation (for existing sub-operations) No need this now because we use only the existing tuples of indices for the variables x and d
        # constraint (7)
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    # if self.instance.jobs_struct[i][j][s] == 1: # if the sub operation exists
                    m.addConstr((gp.quicksum(x[i, j, s, k] for k in range(self.instance.nb_workers)) >= 1), name=f"min_sub_operation_assignment_{i}_{j}_{s}")


        # constraint : C[i] >= d[i, j, k] + processing_time_operations if it is solo, apprentissage or collab
        # Completion time of each job
        # constraint (3)
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for k in range(self.instance.nb_workers):
                        index_s = self.instance.jobs_struct[i][j][s]


                        # print(f"C[{i}] >= d[{i}, {j}, {s}, {k}] + processing_time_sub_op[{i}, {j}, {s}] = {self.instance.processing_time_sub_op[i][j][s]}")
                        
                        # pour le moment self.instance.sub_operations_times[index_s][0] : 0 pour le temps standard pour un worker
                        
                        # m.addConstr((C[i] >= d[i, j, s, k] + self.instance.sub_operations_times[index_s][0] * x[i, j, s, k]), name=f"completion_time_{i}_{j}_{s}_{k}")
                        # m.addConstr((C[i] >= f[i, j, s, k]), name=f"completion_time_{i}_{j}_{s}_{k}")
                        f_ijsk = d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * z_auxilary[i,j,s,0] + self.instance.sub_operations_times[index_s][1] * z_auxilary[i,j,s,1] +self.instance.sub_operations_times[index_s][2] * z_auxilary[i,j,s,2] - M * (1 - x[i,j,s,k])
                        m.addConstr((C[i] >= f_ijsk), name=f"completion_time_{i}_{j}_{s}_{k}")
                        



        # constraint : C_max >= C[i] for all i
        # constraint (4)
        for i in range(self.instance.nb_jobs):
            m.addConstr((C_max >= C[i]), name=f"makespan_{i}")


        # constraint : end_of_sub_op
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    nb_worker_for_ijs = gp.quicksum(x[i, j, s, k] for k in range(self.instance.nb_workers)) # number of workers assigned O_ijs
                    index_s = self.instance.jobs_struct[i][j][s]
                    


        ##### OVERLAP : Hypothèse : PAS DE COLLAB ET TOUTES LES TACHES AFFECTEES
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##### Prise en compte du fait que les workers peuvent collaborer sur une même tache, ajout de z_ijs2 pour cela
        ###### contrainte sur variable auxiliares z
        # nb_pers_O_ijs = 1 * z_ijs0 + 2 * z_ijs1 + 2 * z_ijs2
        # z_ijs0 + z_ijs1 + z_ijs2 = 1
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    
                    # chaque sous-opération est soit en solo, en apprentissage ou en collab
                    m.addConstr((z_auxilary[i,j,s,0] + z_auxilary[i,j,s,1] + z_auxilary[i,j,s,2] == 1), name=f"z_assignment_{i}_{j}_{s}") 
                    
                    # fixé z_ijs0 à 1 si O_ijs est fait en solo, z_ijs1 à 1 si O_ijs est fait en apprentissage, z_ijs2 à 1 si O_ijs est fait en collab
                    m.addConstr(((gp.quicksum(x[i,j,s,k] for k in range(self.instance.nb_workers)) == 1 * z_auxilary[i,j,s,0] + 2 * z_auxilary[i,j,s,1] + 2 * z_auxilary[i,j,s,2])), name=f"x_assignment_{i}_{j}_{s}")
        
                    # si 2 workers alors soit apprentissage soit collab
                    m.addConstr((z_auxilary[i,j,s,1] + z_auxilary[i,j,s,2] <= 1), name=f"if_two_workers_then_apprentissage_or_collab_{i}_{j}_{s}") 
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##### LINEARISATION DU MIN:
        # Level_min = min_{k}{x_ijsk * level_km} avec m = metier de sous-op s

        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                   for k in range(self.instance.nb_workers):
                       index_s = self.instance.jobs_struct[i][j][s]
                       index_m = self.instance.sub_op_to_m[index_s]
                       m.addConstr((Level_min[i,j,s] <= x[i,j,s,k] * self.instance.levels_workers[k][index_m] + M * (1 - x[i,j,s,k])), name=f"linearization_min1_{i}_{j}_{s}_{k}")
                       m.addConstr((Level_min[i,j,s] >= x[i,j,s,k] * self.instance.levels_workers[k][index_m] - M * (1 - Delta_min[i,j,s,k])), name=f"linearization_min2_{i}_{j}_{s}_{k}")
                
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    m.addConstr((gp.quicksum(Delta_min[i,j,s,k] for k in range(self.instance.nb_workers)) == 1), name=f"linearization_min_binary_{i}_{j}_{s}") # Delta_min_ijsk doit prendre 1 pour le k tq il a le level minimal pour cette tache
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #### Utilisation de la variable auxiliaire Level_min pour savoir si la tache est fait en apprentissage ou en collab si fait a deux sinon z_ijs1 = z_ijs2 = 0

        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    index_s = self.instance.jobs_struct[i][j][s]
                    nb_pers_ijs = gp.quicksum(x[i,j,s,k] for k in range(self.instance.nb_workers))
                    m.addConstr((Level_min[i,j,s] + M * z_auxilary[i,j,s,1] >= self.instance.sub_operations_difficulties[index_s]), name=f"level_min_difficulty_beta0_{i}_{j}_{s}")
                    m.addConstr((Level_min[i,j,s] - M * z_auxilary[i,j,s,2] <= self.instance.sub_operations_difficulties[index_s] + M*(2-nb_pers_ijs)), name=f"level_min_difficulty_beta1_{i}_{j}_{s}")

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------




        ##### Hypothèse : PAS DE COLLAB ET TOUTES LES TACHES AFFECTEES
        # constraint (overlap) : d[i,j,s,k] >= f[h,g,,k] - M * delta[i,j,h,g,s,z,k] for all i, j, h, g, s, z, k with (i,j,s) != (h,g,z)
        #                        d[h,g,z,k] >= f[i,j,s,k] - M * (1 - delta[i,j,h,g,s,z,k]) for all i, j, h, g, s, z, k with (i,j,s) != (h,g,z)
        # constraint (5) (6)
        
        for k in range(self.instance.nb_workers):
            for i in range(self.instance.nb_jobs):
                for j in range(len(self.instance.jobs_struct[i])):
                    for h in range(self.instance.nb_jobs):
                        for g in range(len(self.instance.jobs_struct[h])):
                            for s in range(len(self.instance.jobs_struct[i][j])):
                                for z in range(len(self.instance.jobs_struct[h][g])):
                                    if (i, j, s) != (h, g, z):
                                        index_s = self.instance.jobs_struct[i][j][s]
                                        index_z = self.instance.jobs_struct[h][g][z]

                                        f_hgzk = d[h,g,z,k] + self.instance.sub_operations_times[index_z][0] * z_auxilary[h,g,z,0] + self.instance.sub_operations_times[index_z][1] * z_auxilary[h,g,z,1] + self.instance.sub_operations_times[index_z][2] * z_auxilary[h,g,z,2] - M * (1 - x[h, g, z, k])
                                        f_ijsk = d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * z_auxilary[i,j,s,0] + self.instance.sub_operations_times[index_s][1] * z_auxilary[i,j,s,1] + self.instance.sub_operations_times[index_s][2] * z_auxilary[i,j,s,2] - M * (1 - x[i, j, s, k])
                                        

                                        # En mettant == j'ai status.code = 4 de Gurobi (non borné), en mettant >= j'ai status.code = 2 (optimal)
                                        # donc le solveur force f[x1,x2,x3,x4] à être petit
                                        m.addConstr(f[h,g,z,k] >= d[h,g,z,k] + self.instance.sub_operations_times[index_z][0] * z_auxilary[h,g,z,0] + self.instance.sub_operations_times[index_z][1] * z_auxilary[h,g,z,1] + self.instance.sub_operations_times[index_z][2] * z_auxilary[h,g,z,2] - M * (1 - x[h, g, z, k]))
                                        m.addConstr(f[i,j,s,k] >= d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * z_auxilary[i,j,s,0] + self.instance.sub_operations_times[index_s][1] * z_auxilary[i,j,s,1] + self.instance.sub_operations_times[index_s][2] * z_auxilary[i,j,s,2] - M * (1 - x[i, j, s, k]))
                                        


                                        # f_hgzk = d[h,g,z,k] + self.instance.sub_operations_times[index_z][0] * x[h, g, z, k] 
                                        # f_ijsk = d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * x[i, j, s, k]

                                        m.addConstr((d[i,j,s,k] >= f_hgzk - M * delta[i,j,h,g,s,z,k]), name=f"overlap1_{i}_{j}_{h}_{g}_{s}_{z}_{k}")
                                        m.addConstr((d[h,g,z,k] >= f_ijsk - M * (1 - delta[i,j,h,g,s,z,k])), name=f"overlap2_{i}_{j}_{h}_{g}_{s}_{z}_{k}")


        # constraint : if x[i, j, k] = 0 then d[i, j, k] = 0 for all i, j, k
        # contrainte big M pour forcer d[i,j,k] à 0 si x[i,j,k] = 0
        # constraint (10)
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for k in range(self.instance.nb_workers):
                        # print(f"d[{i}, {j}, {s}, {k}] <= M * x[{i}, {j}, {s}, {k}]")
                        m.addConstr((d[i,j,s,k] <= M * x[i,j,s,k]), name=f"start_time_zero_if_not_assigned_{i}_{j}_{s}_{k}")



    ### CONSTRAINTES DE PRECEDENCE
        # constraint : precedence constraints of operations for the same job
        # f_ijk = d_ijk + processing_time_operation_ij * x_ijk
        # f_ijk <= sum_(k' in W) (d_ij'k')
        # constraint (12)
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for j_prime in range(len(self.instance.jobs_struct[i])):
                    for s in range(len(self.instance.jobs_struct[i][j])):
                        for k in range(self.instance.nb_workers):
                            if (j != j_prime) and (self.instance.constraints_precedence_operations[i,j,j_prime] == 1): # if O_ij must be performed before operation O_ij'
                                index_s = self.instance.jobs_struct[i][j][s]


                                # f_ijsk = d[i,j,s,k] + self.instance.processing_time_sub_op[index_s] * x[i,j,s,k] # actif pour 2 workers au maximum # old with same operation time for solo or groups
                                f_ijsk = d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * z_auxilary[i,j,s,0] + self.instance.sub_operations_times[index_s][1] * z_auxilary[i,j,s,1] + self.instance.sub_operations_times[index_s][2] * z_auxilary[i,j,s,2] - M * (1 - x[i,j,s,k])
                                # contrainte juste à la ligne suivante marche si une seul personne est affecté à la sous opération O(ij's) si deux personnes la somme n'a plus d'effet
                                # m.addConstr((f_ijsk <= gp.quicksum(d[i,j_prime,s,k] for k in range(self.instance.nb_workers))), name=f"precedence_operations_{i}_{j}_{j_prime}_{s}_{k}")
                                for k_prime in range(self.instance.nb_workers):
                                    for s_prime in range(len(self.instance.jobs_struct[i][j_prime])):
                                        m.addConstr((f_ijsk <= M * (1 - x[i,j_prime,s_prime,k_prime]) + d[i,j_prime,s_prime,k_prime] ), name=f"precedence_operations_inactive_{i}_{j}_{j_prime}_{s}_{k}")  # contrainte pour désactiver la contrainte de précédence si x[i,j,s,k] = 0



                       
        # constraint : precedence constraints of sub-operations for the same operation
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for s_prime in range(len(self.instance.jobs_struct[i][j])):
                        for k in range(self.instance.nb_workers):
                            if (s != s_prime) and (self.instance.constraints_precedence_sub_operations[i,j,s,s_prime] == 1): # if sub-operation s of operation O_ij must be performed before sub-operation s' of operation O_ij
                                index_s = self.instance.jobs_struct[i][j][s]
                                # f_ijsk = d[i,j,s,k] + self.instance.processing_time_sub_op[index_s] * x[i,j,s,k]
                                f_ijsk = d[i,j,s,k] + self.instance.sub_operations_times[index_s][0] * z_auxilary[i,j,s,0] + self.instance.sub_operations_times[index_s][1] * z_auxilary[i,j,s,1] + self.instance.sub_operations_times[index_s][2] * z_auxilary[i,j,s,2] - M * (1 - x[i,j,s,k])
                                # m.addConstr((f_ijsk <= gp.quicksum(d[i,j,s_prime,k] for k in range(self.instance.nb_workers))), name=f"precedence_sub_operations_{i}_{j}_{s}_{s_prime}_{k}")
                                for k_prime in range(self.instance.nb_workers):
                                    # for s_prime in range(self.instance.max_nb_sub_operations):
                                    m.addConstr((f_ijsk <= M * (1 - x[i,j,s_prime,k_prime]) + d[i,j,s_prime,k_prime] ), name=f"precedence_sub_operations_inactive_{i}_{j}_{s}_{s_prime}_{k}")  # contrainte pour désactiver la contrainte de précédence si x[i,j,s,k] = 0
        

        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    index_s = self.instance.jobs_struct[i][j][s]
                    if self.instance.sub_operations_times[index_s][2] == -1 : # if collab is not possible for this sub-op
                        m.addConstr((z_auxilary[i,j,s,2] == 0), name=f"no_collab_{i}_{j}_{s}")

        # constraint : if worker start the first sub-operation of an operation of job then this worker must do all the operation (all sub-operations of the operation) for this job
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for k in range(self.instance.nb_workers):
                        # if self.instance.jobs_struct[i][j][s] == 1: # if the sub operation exists
                        # print(f"x[{i}, {0}, {0}, {k}] <= x[{i}, {j}, {s}, {k}]")
                        m.addConstr((x[i,0,0,k] <= x[i,j,s,k]), name=f"same_worker_operation_{i}_{j}_{s}_{k}")


        # constraint : level of worker k must be >= difficulty of the sub operation assigned to worker k or if 2 workers are asssigned to the same sub-op, at least one of the two workers must have a level higher than the difficulty of the sub-op
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for k in range(self.instance.nb_workers):
                        index_s = self.instance.jobs_struct[i][j][s]
                        # recuperer de quelle corps de métier correspond cette sub-op
                        index_m = self.instance.sub_op_to_m[index_s]
                        if self.instance.levels_workers[k][index_m] < self.instance.sub_operations_difficulties[index_s]:
                            # m.addConstr((x[i,j,s,k] * self.instance.levels_workers[k][index_s] >= self.instance.difficulty_sub_op[index_s] * x[i,j,s,k]), name=f"worker_with_capacity{i}_{j}_{s}_{k}") 
                            m.addConstr((x[i,j,s,k] + gp.quicksum(x[i,j,s,k_prime] for k_prime in range(self.instance.nb_workers) if k_prime != k) >= M*(x[i,j,s,k] - 1) + 2), name=f"at_most_two_workers_{i}_{j}_{s}_{k}") # at most 2 workers can be assigned to the same sub-op if k do the sub op and he dont have the levels for it
                            
                            # La contrainte suivante permet de modéliser que si cette tache est fait par x_ijs et n'a pas le niveau alors l'autre personne avec lui doit l'avoir
                            m.addConstr((gp.quicksum(x[i,j,s,k_prime] * self.instance.levels_workers[k_prime][index_m] for k_prime in range(self.instance.nb_workers) if k_prime != k)) >= self.instance.sub_operations_difficulties[index_s] * x[i,j,s,k], name=f"at_least_one_worker_with_capacity_{i}_{j}_{s}_{k}") # if worker k do the sub op and he dont have the levels for it, at least one of the other workers assigned to the same sub-op must have the level for it
        
        # # constraint : count number of sub-operations assigned to each worker k and each sub-operation s
        # # I try this for the learning effect
        # count = [ [0 for s in range(self.instance.nb_sub_operations)] for k in range(self.instance.nb_workers) ] # count[k][s] = number of sub-operations assigned to worker k
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         for s in range(len(self.instance.jobs_struct[i][j])):
        #             for k in range(self.instance.nb_workers):
        #                 index_s = self.instance.jobs_struct[i][j][s]
        #                 count[k][index_s] += x[i,j,s,k]

        # for k in range(self.instance.nb_workers):
        #     for s in range(self.instance.nb_sub_operations):
        #         m.addConstr((y[k,s] == count[k][s]), name=f"count_sub_operations_{k}_{s}")

                

        # constraint : if two worker k1 and k2 are assigned to the same sub-operation (i,j,s) then the starting time of the sub-operation for both workers must be the same : d[i,j,s,k1] = d[i,j,s,k2]
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for s in range(len(self.instance.jobs_struct[i][j])):
                    for k1 in range(self.instance.nb_workers):
                        for k2 in range(k1+1, self.instance.nb_workers): # k2 > k1 to avoid duplicate constraints
                            # print(f"d[{i}, {j}, {s}, {k1}] <= d[{i}, {j}, {s}, {k2}] + M * (2 - x[{i}, {j}, {s}, {k1}] + x[{i}, {j}, {s}, {k2}])")
                            # print(f"d[{i}, {j}, {s}, {k2}] <= d[{i}, {j}, {s}, {k1}] + M * (2 - x[{i}, {j}, {s}, {k1}] + x[{i}, {j}, {s}, {k2}])")
                            m.addConstr((d[i,j,s,k1] <= d[i,j,s,k2] + M * (2 - (x[i,j,s,k1] + x[i,j,s,k2]))), name=f"same_start_time1_{i}_{j}_{s}_{k1}_{k2}")
                            m.addConstr((d[i,j,s,k2] <= d[i,j,s,k1] + M * (2 - (x[i,j,s,k1] + x[i,j,s,k2]))), name=f"same_start_time2_{i}_{j}_{s}_{k1}_{k2}")


        # l -> l[worker][m] niveau du worker pour sous op de métier m 
        # level of worker k for sub-op s after learning effect can not be more than 1 unit higher than the initial level of worker k for sub-op s
        # constraint learning effect
        for k in range(self.instance.nb_workers):
            for metier in range(self.instance.nb_professions): # We fix a worker and a metier
                # print(f"w_{k}  --  metier {metier}")
                
                all_sub_op_m_learning = 0 # number of times worker k do the sub_op in m
                for i in range(self.instance.nb_jobs):
                    for j in range(len(self.instance.jobs_struct[i])):
                        for s in range(len(self.instance.jobs_struct[i][j])):
                            index_s = self.instance.jobs_struct[i][j][s]
                            # print(f"s = {s}, index_s = {index_s}")
                            if self.instance.sub_op_to_m[index_s] == metier and self.instance.levels_workers[k][metier] < self.instance.sub_operations_difficulties[index_s]: # Si sous-op s du metier metier assigné aux worker et qu'il n'avait pas le niveau -> apprentissage
                                all_sub_op_m_learning += x[i,j,s,k] # number of times worker k do the sub_op in metier
                                # print(f"index_s = {index_s}, s = {s}, metier = {metier}, sub_op_to_m = {self.instance.sub_op_to_m[index_s]}")
                                # print(f"all_sub_op_m_learning = {all_sub_op_m_learning}")
                                # print(f"l[{k}, {metier}] <= {self.instance.levels_workers[k][metier]} + {all_sub_op_m_learning}*0.5")
                                # print(f"l[{k}, {metier}] <= {self.instance.levels_workers[k][metier]} + 1")
                                # print(f"l[{k}, {metier}] <= 4")

                m.addConstr((l[k,metier] <= self.instance.levels_workers[k][metier] + all_sub_op_m_learning*0.5), name=f"learning_effect_w{k}_metier{metier}") # learning effect for metier metier
                # contrainte suivante peut etre omis ?
                m.addConstr((l[k,metier] <= self.instance.levels_workers[k][metier] + 1), name=f"max_learning_effect_w{k}_metier{metier}") # max learning effect for metier metier
                m.addConstr((l[k,metier] >= self.instance.levels_workers[k][metier]), name=f"min_level_metier{metier}") # level of worker k for metier metier can not be less than the initial level of worker k for metier metier
                m.addConstr((l[k,metier] <= 4), name=f"max_level_metier{metier}") # level of worker k for metier metier can not be more than 4 because the max difficulty of sub-op is 4
                        
            
            
            # for s in range(len(self.instance.difficulty_sub_op)): # We fix a worker and a sub_op
            #     print(f"w_{k}  --  sous op {s}")
                
            #     all_sub_op_m = 0 # number of times worker k do the sub_op in m
            #     for i in range(self.instance.nb_jobs):
            #         for j in range(len(self.instance.jobs_struct[i])):
            #             for s in range(len(self.instance.jobs_struct[i][j])):
            #                 index_s = self.instance.jobs_struct[i][j][s]
            #                 if index_sub_op == s :
            #                     all_sub_op_s += x[i,j,sub_op,k] # number of times worker k do the sub_op s
            #                     print(f"index_sub_op = {index_sub_op}, s = {s}, sub_op = {sub_op}")
                                
            #     # savoir toute les opération de m que le worker à éffectué auquel il n'avait pas la compétence
            #     m.addConstr((l[k,m]) == self.instance.levels_workers[k][m] + all_sub_op_m*0.5)


            #     m.addConstr((l[k,s] <= self.instance.levels_workers[k][s] + all_sub_op_s*1.1 ), name=f"learning_effect_w{k}_subop{s}") # learning effect
            #     m.addConstr((l[k,s] >= self.instance.levels_workers[k][s]), name=f"min_level_w{k}_subop{s}") # level of worker k for sub-op s can not be less than the initial level of worker k for sub-op s
            #     m.addConstr((l[k,s] <= self.instance.levels_workers[k][s] + 1), name=f"max_level_w{k}_subop{s}") # max level of worker k for sub-op s can not be more than 1 unit higher than the initial level of worker k for sub-op s
            #     m.addConstr((l[k,s] <= 4), name=f"max_level_sub_op{s}") # level of worker k for sub-op s can not be more than the difficulty of sub-op s



        # # constraint :
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         for s in range(len(self.instance.jobs_struct[i][j])):
        #             for k in range(self.instance.nb_workers):


        # m.addConstr((C_max <= 60), name="max_makespan")


        ########################################################################
        ########################### OBJECTIVE FUNCTION #########################
        ########################################################################
        
        if objective == "makespan":
            m.setObjective(C_max, GRB.MINIMIZE)

        else :
            skill_obj = gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
            
            if objective == "skill":
                m.setObjective(skill_obj, GRB.MAXIMIZE)

            elif objective == "lexicographic":
                m.setObjectiveN(C_max, index=0, priority=priority[0], name="minimize_makespan")
                m.setObjectiveN(-skill_obj, index=1, priority=priority[1], name="maximize_skill_levels")
                m.modelSense = GRB.MINIMIZE

            elif objective == "both":
                m.setObjectiveN(C_max, index=0, weight=weight[0], name="minimize_makespan")
                m.setObjectiveN(-skill_obj, index=1, weight=weight[1], name="maximize_skill_levels")
                m.modelSense = GRB.MINIMIZE
            


        # sum_Ci_obj = gp.quicksum(C[i] for i in range(self.instance.nb_jobs))

        # # minimsier somme des complétudes des jobs
        # m.setObjective(sum_Ci_obj, GRB.MINIMIZE)

        # # double objectif : minimiser le makespan et maximiser le niveau de compétence des travailleurs
        # m.setObjective(0.5 * C_max - 0.5 * skill_obj, GRB.MINIMIZE)

        m.write(f"../results/model_{objective}.lp")
        return m





    def solve(self, objective="makespan", weight=[0,0], priority=[0,1], verbose=False):
        """
        Résout le modèle et affiche les résultats
        
        Args:
            objective (str): l'objectif à optimiser, peut être "makespan", "skill", "both" ou "lexicographic"
            weight (list): les poids à accorder à chaque objectifs (makespan, skill) si objective = "both", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (makespan, skill) si objective = "lexicographic", n'est pas utilisé sinon
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            
        Returns:
            (Solution): une instance de la classe Solution contenant les résultats de la résolution du modèle
        """

        assert objective in ["makespan", "skill", "both", "lexicographic"], "objective doit être 'makespan', 'skill', 'both' ou 'lexicographic'"
        assert len(weight) == 2, "weight doit être une liste de deux éléments"
        if objective == "both":
            assert sum(weight) == 1, "la somme des poids doit être égale à 1"
        assert len(priority) == 2, "priority doit être une liste de deux éléments"
    
        m = self._build_model(objective, weight, priority)

        if verbose == False:
            m.setParam('OutputFlag', 0) # to disable gurobi output

        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found with objective value:", m.objVal)
            m.write("../results/solution.sol")
        else:
            print("No optimal solution found. Status code:", m.status)
            return


        if verbose:
            # Query number of multiple objectives, and number of solutions
            nSolutions = m.SolCount
            nObjectives = m.NumObj
            print("Problem has", nObjectives, "objectives")
            print("Gurobi found ", nSolutions, "solutions")

            if nObjectives > 1:
                solutions = []
                for s in range(nSolutions):
                    # Set which solution we will query from now on
                    m.params.SolutionNumber = s

                    # Print objective value of this solution in each optimization pass
                    print('\nSolution', s, ':', end='')
                    for o in range(nObjectives):
                        # Set which objective we will query
                        m.params.ObjNumber = o
                        # Query the objective value for the corresponding optimization pass
                        print('  Obj', o, '=', m.ObjNVal, end='')
                    # Print first three variables in the solution
                print("\n***********************")
                print("All values:")
                print(m.objNVal)

            else:
                print("Objective value:", m.objVal)
                for s in range(nSolutions):
                    m.params.SolutionNumber = s
                    print('\nSolution', s, ':', end='')
                    print('  Obj =', m.ObjVal, end='')
                print("\n------------------")


        ##### build solution object
        all_vars = m.getVars()
        # print("len(all_vars)", len(all_vars))
        values = m.getAttr('X', all_vars)
        names = m.getAttr('VarName', all_vars)
        res = []
        for name, value in zip(names, values):
            res.append((name, value))
        
        if verbose :
            print("objective value:", m.objVal)
            # print(res)
        return Solution(res, self.instance)    

    
class Solution:
    def __init__(self, var_list, instance):
        self.x = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.nb_workers)) # x[i, j, s, k] = 1 if sub operation s of  operation j of job i is assigned to worker k, 0 otherwise
        self.d = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.nb_workers))
        self.C = np.zeros(instance.nb_jobs)
        self.C_max = 0
        self.delta = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.max_nb_sub_operations, instance.nb_workers))
        self.l = np.zeros((instance.nb_workers, instance.nb_professions))
        self.f = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, instance.nb_workers))
        self.z_auxilary = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.max_nb_sub_operations, 3)) # z_auxilary[i,j,s,z] = 1 if sub-op s of operation j of job i is done en solo (z=0) ou en apprentissage (z=1) ou en collab (z=2)


        # print("var_list", var_list)
        for v in var_list:

            if v[0][0][0] == "x":
                indices = v[0][2:-1].split(",") # x[i, j, s, k] -> indices = [i, j, s, k]
                i, j, s, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
                # print(f"x[{i}, {j}, {s}, {k}] = {v[1]}")
                self.x[i, j, s, k] = v[1]

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

            elif v[0][0] == "l" : # l[k, m] -> indices = [k, m]
                indices = v[0][2:-1].split(",")
                k, m = int(indices[0]), int(indices[1])
                self.l[k, m] = v[1]

            elif v[0][0] == "f": # f[i, j, s, k] -> indices = [i, j, s, k]
                indices = v[0][2:-1].split(",")
                i, j, s, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
                self.f[i, j, s, k] = v[1]

            elif v[0][:10] == "z_auxilary": # z_auxilary[i, j, s, z] -> indices = [i, j, s, z]
                indices = v[0][11:-1].split(",")
                i, j, s, z = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
                self.z_auxilary[i, j, s, z] = v[1]
            

    def __str__(self):
        res = (f"x: {self.x.shape} \n{self.x}\n"
               f"d: {self.d.shape} \n{self.d}\n"
               f"C: {self.C.shape} \n{self.C}\n"
               f"C_max: {self.C_max}\n"
            #    f"delta: {self.delta.shape}\n{self.delta}\n"
               f"l: {self.l.shape}\n{self.l}\n")
        return res


if __name__ == "__main__":
    # nb_jobs, max_nb_operations, nb_sub_operations,  max_nb_sub_operations, sub_operations_times, sub_operations_difficulties, nb_workers, levels_workers, difficulty_jobs, jobs_struct, constraints_precedence_operations, constraints_precedence_sub_operations = read_file("../data/data_temp.test")
    res = read_file("../data/data_temp.test")
    # instance = Instance(nb_jobs, max_nb_operations, nb_sub_operations, max_nb_sub_operations, sub_operations_times, sub_operations_difficulties, nb_workers, levels_workers, difficulty_jobs, jobs_struct, constraints_precedence_operations, constraints_precedence_sub_operations)
    instance = Instance(res)

    print(instance)
    model = Model(instance)
    s = model.solve(objective="both", weight=[0.5, 0.5], priority=[0, 1], verbose=True)
    print(s)

    gantt_chart(s, instance, color=0)
    plot_levels_workers(s, instance, verbose=True)