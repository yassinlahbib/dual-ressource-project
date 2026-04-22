import gurobipy as gp
from gurobipy import GRB
from utils import *
import numpy as np
import time

class Instance:
    def __init__(self):
        # VOIR si une maniere plus simple pour instancier l'instance
        pass

    def from_dictionary(self, dictionary):
        # INT
        self.nb_jobs = dictionary["nb_jobs"] # int
        self.nb_professions = dictionary["nb_professions"] # int
        self.nb_task_in_profession = dictionary["nb_task_in_profession"] # size (nb_professions)
        self.max_nb_operations = dictionary["max_nb_operations"] # int
        self.nb_tasks = dictionary["nb_tasks"] # int
        self.nb_workers = dictionary["nb_workers"] # int

        # DIFFICULTIES AND TIMES
        self.tasks_difficulties = dictionary["tasks_difficulties"] # size (nb_tasks)
        self.tasks_times = dictionary["tasks_times"] # size (nb_tasks, 3)

        # LEVELS OF WORKERS
        self.levels_workers = dictionary["levels_workers"] # size (nb_workers, nb_professions)
        # Forgetting effect
        # self.forgetting = dictionary["forgetting_workers"] # size (nb_workers, nb_professions)
        
        # JOBS STRUCTURE
        self.jobs_struct = dictionary["jobs_struct"] # len=nb_jobs, len(jobs_struct[i]) = number of operations of job i.
        self.difficulty_jobs = dictionary["difficulty_jobs"] # size (nb_jobs)

        # CONSTRAINTS
        self.constraints_precedence_operations = dictionary["constraints_precedence_operations"] # size(nb_jobs, max_nb_operations, max_nb_operations)
        
        # MAPPING TASK TO METIER
        self.task_to_m = dictionary["dict_task_to_m"]


    def from_random(self, at_least_a_worker_have_competence_for_each_profession=True, seed=42):
        
        np.random.seed(seed)
        
        self.nb_jobs = np.random.randint(2, 4)
        self.nb_professions = np.random.randint(3, 5)
        
        self.nb_task_in_profession = np.random.randint(2, 4, size=self.nb_professions)
        self.nb_tasks = np.sum(self.nb_task_in_profession)

        self.max_nb_operations = min(np.random.randint(4, 6), self.nb_tasks)
        self.nb_workers = np.random.randint(3, 7)
        

        time_solo_tasks = np.random.randint(1, 20, size=self.nb_tasks)
        time_teaching_tasks = time_solo_tasks * np.random.uniform(1.2, 1.5, size=self.nb_tasks)
        time_collab_tasks = time_solo_tasks * np.random.uniform(0.5, 1.0, size=self.nb_tasks)

        # certaines taches ne peuvent pas être fait en collab
        for i in range(self.nb_tasks):
            if np.random.rand() < 0.1:
                time_collab_tasks[i] = -1

        self.tasks_times = np.stack((time_solo_tasks, time_teaching_tasks, time_collab_tasks), axis=1)
        self.tasks_difficulties = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=self.nb_tasks)

        self.task_to_m = {}
        id_task = np.arange(np.sum(self.nb_task_in_profession))
        for m in range(self.nb_professions):
            for t in range(self.nb_task_in_profession[m]):
                self.task_to_m[id_task[0]] = m
                id_task = id_task[1:]

        # par metier il faut au moins un worker avec la compétence nécessaire pour faire les taches de ce metier sinon pas de solution pour l'instance (en attendant de faire soft contraintes sur cela)
        max_difficulty_per_profession = np.zeros(self.nb_professions)

        for metier in range(self.nb_professions):
            max_diff = 0
            for t in range(self.nb_tasks):
                if self.task_to_m[t] == metier:
                    if self.tasks_difficulties[t] > max_diff:
                        max_diff = self.tasks_difficulties[t]
            max_difficulty_per_profession[metier] = max_diff

        self.levels_workers = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1-1, size=(self.nb_workers, self.nb_professions))


        self.jobs_struct = []
        for i in range(self.nb_jobs):
            nb_operations = np.random.randint(3, self.max_nb_operations)
            operations = np.random.choice(self.nb_tasks, size=nb_operations, replace=True) #on peut avoir des taches qui se répètent dans le même job
            self.jobs_struct.append(operations)
        
        self.difficulty_jobs = np.array([max(self.tasks_difficulties[operations]) for operations in self.jobs_struct])


        self.constraints_precedence_operations = np.zeros((self.nb_jobs, self.max_nb_operations, self.max_nb_operations))
        for i in range(self.nb_jobs):
            for j in range(len(self.jobs_struct[i]) - 1):
                self.constraints_precedence_operations[i,j,j+1] = 1
                # for j_prime in range(j+1, len(self.jobs_struct[i])):


        # Pour pouvoir suivre le phénomène d'oublis des workers par corps de métier
        # self.forgetting = np.zeros((self.nb_workers, self.nb_professions)) # initialisé à 0 pour le départ

        if at_least_a_worker_have_competence_for_each_profession: # Au moins 1 worker à le niveau de compétence de la tache la plus difficile
            self.levels_workers = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=(self.nb_workers, self.nb_professions))
            for metier in range(self.nb_professions):
                if np.max(self.levels_workers[:, metier]) < max_difficulty_per_profession[metier]:
                    worker_to_change = np.random.randint(0, self.nb_workers)
                    self.levels_workers[worker_to_change, metier] = max_difficulty_per_profession[metier]

    def qualified_workers_for_task(self, verbose=False):
        # fonction pour visualisation les indices commence à 1  pour les workers
        res  = []
        for i in range(len(self.jobs_struct)):
            res.append([])
            for j in range(len(self.jobs_struct[i])):
                res[i].append([])
        for i in range(len(self.jobs_struct)):
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                level_task = self.tasks_difficulties[index_task] # niveau de diificulté de la tache 
                m = self.task_to_m[index_task]
                # print(f"tache {index_task} - niveau {level_task} - metier {m}")
                for k in range(self.nb_workers):
                    # print(f"\tworker {k+1} - level {self.levels_workers[k][m]}")
                    if self.levels_workers[k][m] >= level_task:
                        res[i][j].append(k+1) # k+1 pour visualisation on commence les worker par w_1

        if verbose == True:
            for i in range(len(self.jobs_struct)):
                print(f"J{i+1}: ",end="")
                for j in range(len(self.jobs_struct[i])):
                    print(f" ({i+1},{j+1}): {res[i][j]} \t", end="")
                print("\n")
                    
        return res





        
    def __str__(self):

        jobs_struct_str = ""
        for i in range(self.nb_jobs):
            jobs_struct_str += f"Job {i} : "
            for j in range(len(self.jobs_struct[i])):
                jobs_struct_str += f"\tO_({i},{j}) = {self.jobs_struct[i][j]} "
            jobs_struct_str += "\n"

        res =  (f"\n ===== Start of Instance: =====\n"
               f"Number of jobs: {self.nb_jobs}\n"
               f"Number of professions: {self.nb_professions}\n"
               f"Number of tasks per profession: {self.nb_task_in_profession}\n"
               f"Max number of operations per Jobs: {self.max_nb_operations}\n"
               f"Total number of tasks: {self.nb_tasks}\n"
               f"Total number of workers: {self.nb_workers}\n"
               f"Task to profession mapping:\n{self.task_to_m}\n"


               f"Worker levels: shape={self.levels_workers.shape}\n {self.levels_workers}\n"
            #    f"Worker forgetting: shape={self.forgetting.shape}\n{self.forgetting}\n" 
               f"Job difficulties: shape= {self.difficulty_jobs.shape}\n{self.difficulty_jobs}\n"
               f"Task difficulties: shape= {self.tasks_difficulties.shape}\n{self.tasks_difficulties}\n"
               f"Task processing times: shape= {self.tasks_times.shape}\n{self.tasks_times}\n")
        
        res += (f"Jobs structure: len= {len(self.jobs_struct)}\n{jobs_struct_str}\n"
               f"Precedence constraints: shape= {self.constraints_precedence_operations.shape}\n{self.constraints_precedence_operations}\n"
               f"\n ===== End of Instance: =====\n")
        return res

COEF_LEARNING = 0.5
COEF_TUTOR = 0.7
COEF_APPRENTI = 0.3
COEF_FORGETTING = 0.1
COEF_COLLAB = 0.5
LEVEL_MIN = 1
LEVEL_MAX = 4
EPS = 0.01
M = 10000
# temps de la période courante considéré par le PL_i
BORN_SUP_MAKESPAN = 20 # on se fixe une durée de 20 unités de temps pour réaliser les taches sur la période courante. Les taches qui dépassent cette duré seront considéré comme une pénalité dans le calcul du makespan et pourront être traité dans une autre période
PERC_SOLO_NO_LEVEL_TIME = 1.50 #pourcentage de temps en plus pour les taches en solo sans le level requis



class Model:
    def __init__(self, instance):
        self.instance = instance

    def write_objectives_values(self, m, nObjectives, file_name):
        """
        Ecrit la valeur de chaque objectif dans un fichier texte
        Args:
            m (gurobi.Model): le modèle gurobi après optimisation
            nObjectives (int): le nombre d'objectifs du modèle
            file_name (str): le nom du fichier dans lequel écrire les valeurs des objectifs
        Returns:
            None
        """
        with open(file_name, 'w') as f:
            print("ici -> ", nObjectives)
            if nObjectives == 1:
                f.write(f"Obj: {m.ObjVal}\n")
            else: 
                for o in range(nObjectives):
                    m.params.ObjNumber = o
                    f.write(f"Obj{o}: {m.ObjNVal}\n")
        f.close()

    def _add_physical_constraints(self, m, x, d, C, C_max, z_auxilary, Level_min, Delta_min, f, delta):
    
        # constraint :
        # At most 2 workers per tasks
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])): # for all operations except the first one of each job
                m.addConstr((gp.quicksum(x[i, j, k] for k in range(self.instance.nb_workers)) <= 2), name=f"max_sub_operation_assignment_{i}_{j}")
        

        # constraint :
        # All operations must be assigned to at least one worker
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((gp.quicksum(x[i, j, k] for k in range(self.instance.nb_workers)) >= 1), name=f"min_sub_operation_assignment_{i}_{j}")


        # constraint : C[i] >= d[i, j, k] + processing_time_operations
        # Completion time of each job
        # if we minimize makespan, we can have all C[i] = C_max, if we minimize sum of C[i], C[i] will be the completion time of job i
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    M = 0
                    f_ijk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] +self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i,j,k])
                    m.addConstr((C[i] >= f_ijk), name=f"completion_time_{i}_{j}_{k}")
    

        # constraint : C_max >= C[i] for all i
        for i in range(self.instance.nb_jobs):
            m.addConstr((C_max >= C[i]), name=f"makespan_{i}")


        # constraint : auxilary variables z_ijs0, z_ijs1, z_ijs2 do not take value of 1 for the same operation and fixed according to the number of workers assigned to the operation
        
        # ########### OLD -> HARD CONSTRAINTS ##########
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):  
        #         # chaque sous-opération est soit en solo, en apprentissage ou en collab
        #         m.addConstr((z_auxilary[i,j,0] + z_auxilary[i,j,1] + z_auxilary[i,j,2] == 1), name=f"z_assignment_{i}_{j}") 
        #         # fixé z_ijs0 à 1 si O_ijs est fait en solo, z_ijs1 à 1 si O_ijs est fait en apprentissage, z_ijs2 à 1 si O_ijs est fait en collab
        #         m.addConstr(((gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers)) == 1 * z_auxilary[i,j,0] + 2 * z_auxilary[i,j,1] + 2 * z_auxilary[i,j,2])), name=f"x_assignment_{i}_{j}")
        #         # si 2 workers alors soit apprentissage soit collab
        #         m.addConstr((z_auxilary[i,j,1] + z_auxilary[i,j,2] <= 1), name=f"if_two_workers_then_apprentissage_or_collab_{i}_{j}") 

        ########### NEW -> SOFT CONSTRAINTS with penalty in the objective function if not respected ##########
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):  
                # chaque sous-opération est soit en solo, en apprentissage ou en collab
                m.addConstr((z_auxilary[i,j,0] + z_auxilary[i,j,1] + z_auxilary[i,j,2] + z_auxilary[i,j,3] == 1), name=f"z_assignment_{i}_{j}") 
                # fixé z_ijs0 à 1 si O_ijs est fait en solo_level ou solo sans level.   z_ijs1 à 1 si O_ijs est fait en apprentissage, z_ijs2 à 1 si O_ijs est fait en collab
                m.addConstr(((gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers)) == 1 * z_auxilary[i,j,0] + 1 * z_auxilary[i,j,3] + 2 * z_auxilary[i,j,1] + 2 * z_auxilary[i,j,2])), name=f"x_assignment_{i}_{j}")
                # si 2 workers alors soit apprentissage soit collab
                # m.addConstr((z_auxilary[i,j,1] + z_auxilary[i,j,2] <= 1), name=f"if_two_workers_then_apprentissage_or_collab_{i}_{j}") 


        # constraint : LINEARISATION of MIN
        # Level_min = min_{k}{x_ijsk * level_km} with m = metier of O_ij

        #================ explanation of linearization of min constraints: ================
        # if Delta_min_ijk = 1 for a worker k' not assigned to O_ij with level k' > level of worker k assigned to O_ij with level_min
        # then constraint is not respected
        
        # if Delta_min_ijk = 1 for a worker k' not assigned to O_ij with level k' <= level of worker k assigned to O_ij with level_min
        # then like is not assigned to the operation Level_min <=0 and >= level of k' -> not possible
        
        # So Delta_min_ijk must be 1 for the worker k' with the level min assigned to O_ij 

        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    # Level_min_ij <= min of levels of workers assinged to O_ij
                    M = LEVEL_MAX
                    m.addConstr((Level_min[i,j] <= x[i,j,k] * self.instance.levels_workers[k][index_m] + M * (1 - x[i,j,k])), name=f"linearization_min1_{i}_{j}_{k}")
                    # Level_min_ij >= min of levels of workers assinged to O_ij
                    m.addConstr((Level_min[i,j] >= x[i,j,k] * self.instance.levels_workers[k][index_m] - M * (1 - Delta_min[i,j,k])), name=f"linearization_min2_{i}_{j}_{k}")

        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((gp.quicksum(Delta_min[i,j,k] for k in range(self.instance.nb_workers)) == 1), name=f"linearization_min_binary_{i}_{j}") # Delta_min_ijsk doit prendre 1 pour le k tq il a le level minimal pour cette tache
        # forcer que Delta_min[ijk] peut prendre la valeur de 1 que pour un worker affécté à la tache
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((Delta_min[i,j,k] <= x[i,j,k]), name=f"linearization_min_binary2_{i}_{j}_{k}") # Delta_min_ijsk doit être égal à 0 si le worker k n'est pas assigné à la tache (i,j)

        # constraint : to know the mode of an operation (solo, apprentissage or collab) with the variable Level_min to fixe auxilary variables z_ijs0, z_ijs1, z_ijs2
        M = LEVEL_MAX
        
        # ########## OLD -> HARD CONSTRAINTS ##########
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         index_j = self.instance.jobs_struct[i][j]
        #         nb_pers_ij = gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))

        #         m.addConstr((Level_min[i,j] + M * z_auxilary[i,j,1] >= self.instance.tasks_difficulties[index_j]), name=f"level_min_difficulty_beta0_{i}_{j}")
        #         m.addConstr((Level_min[i,j] - M * z_auxilary[i,j,2] <= self.instance.tasks_difficulties[index_j] - EPS + M*(2-nb_pers_ij)), name=f"level_min_difficulty_beta1_{i}_{j}")

        ########### NEW -> SOFT CONSTRAINTS with penalty in the objective function if not respected ##########
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
   
                    m.addConstr((self.instance.levels_workers[k][index_m] - self.instance.tasks_difficulties[index_j] * x[i, j, k] >=  - M * z_auxilary[i, j, 3] - M * z_auxilary[i, j, 1]), name=f"level_min_difficulty_beta0_{i}_{j}_{k}")
                    m.addConstr((self.instance.levels_workers[k][index_m] - (self.instance.tasks_difficulties[index_j] - EPS) * x[i,j,k] <=  (1 - x[i,j,k]) * M + (M * z_auxilary[i,j,0]) + (M * z_auxilary[i,j,1]) + (M * z_auxilary[i,j,2])), name=f"level_min_difficulty_beta1_{i}_{j}_{k}")

        # ########## OLD -> HARD CONSTRAINTS ##########
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         index_j = self.instance.jobs_struct[i][j]
        #         nb_pers_ij = gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))

        #         m.addConstr((Level_min[i,j] + M * z_auxilary[i,j,1] >= self.instance.tasks_difficulties[index_j]), name=f"level_min_difficulty_beta0_{i}_{j}")
        #         m.addConstr((Level_min[i,j] - M * z_auxilary[i,j,2] <= self.instance.tasks_difficulties[index_j] - EPS + M*(2-nb_pers_ij)), name=f"level_min_difficulty_beta1_{i}_{j}")

        ########## NEW -> SOFT CONSTRAINTS with penalty in the objective function if not respected ##########
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                index_j = self.instance.jobs_struct[i][j]
                nb_pers_ij = gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))

                m.addConstr((Level_min[i,j] + M * z_auxilary[i,j,1] + M * z_auxilary[i,j,3] >= self.instance.tasks_difficulties[index_j]), name=f"level_min_difficulty_beta2_{i}_{j}")
                m.addConstr((Level_min[i,j] - M * z_auxilary[i,j,2] <= self.instance.tasks_difficulties[index_j] - EPS + M*(2-nb_pers_ij)), name=f"level_min_difficulty_beta3_{i}_{j}")


        # Probleme sur les variable de f [i,j,k] elles sont supérieur à la fin de la date de debut + temps process 
        # -> s'il fait la tache f <= d + times
        # si il fait pas f <= 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    time = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3] * PERC_SOLO_NO_LEVEL_TIME)
                    m.addConstr((f[i,j,k] <= time), name=f"definition_f_{i}_{j}_{k}") # s'il fait la tache f <= d + times, s'il la fait pas f est libre AVOIR SI DERANGEANT
                    # contraintes a verif !!!!!!
                    m.addConstr((f[i,j,k] >= time - M * (1 - x[i,j,k])), name=f"definition_f2_{i}_{j}_{k}") 

        # constraint : OVERLAP
        M = 1000
        for k in range(self.instance.nb_workers):
            for i in range(self.instance.nb_jobs):
                for j in range(len(self.instance.jobs_struct[i])):
                    for h in range(self.instance.nb_jobs):
                        for g in range(len(self.instance.jobs_struct[h])):
                            if (i, j) != (h, g):
                                index_j = self.instance.jobs_struct[i][j]
                                index_g = self.instance.jobs_struct[h][g]

                                ## !!!!!!!!!!!!!!!!
                                ## !!!!!!!!!!!!!!!!
                                # En mettant == j'ai status.code = 4 de Gurobi (non borné), en mettant >= j'ai status.code = 2 (optimal)
                                # donc le solveur force f[x1,x2,x3,x4] à être petit
                                m.addConstr(f[h,g,k] >= d[h,g,k] + self.instance.tasks_times[index_g][0] * z_auxilary[h,g,0] + self.instance.tasks_times[index_g][1] * z_auxilary[h,g,1] + self.instance.tasks_times[index_g][2] * z_auxilary[h,g,2] + (self.instance.tasks_times[index_g][0] * z_auxilary[h,g,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[h, g, k]))
                                m.addConstr(f[i,j,k] >= d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i, j, k]))
                                
                                f_hgk = d[h,g,k] + self.instance.tasks_times[index_g][0] * z_auxilary[h,g,0] + self.instance.tasks_times[index_g][1] * z_auxilary[h,g,1] + self.instance.tasks_times[index_g][2] * z_auxilary[h,g,2] + (self.instance.tasks_times[index_g][0] * z_auxilary[h,g,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[h, g, k])
                                f_ijk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i, j, k])
                                
                                m.addConstr((d[i,j,k] >= f_hgk - M * delta[i,j,h,g,k]), name=f"overlap1_{i}_{j}_{h}_{g}_{k}")
                                m.addConstr((d[h,g,k] >= f_ijk - M * (1 - delta[i,j,h,g,k])), name=f"overlap2_{i}_{j}_{h}_{g}_{k}")



        # constraint : if x[i, j, k] = 0 then d[i, j, k] = 0 for all i, j, k
        # contrainte big M pour forcer d[i,j,k] à 0 si x[i,j,k] = 0
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((d[i,j,k] <= M * x[i,j,k]), name=f"start_time_zero_if_not_assigned_{i}_{j}_{k}")

        
        # constraint : PRECEDENCE of operation of the same job
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for j_prime in range(len(self.instance.jobs_struct[i])):
                    for k in range(self.instance.nb_workers):
                        if (j != j_prime) and (self.instance.constraints_precedence_operations[i,j,j_prime] == 1): # if O_ij must be performed before operation O_ij'
                            index_j = self.instance.jobs_struct[i][j]

                            f_ijk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i,j,k])
                            for k_prime in range(self.instance.nb_workers):
                                    m.addConstr((f_ijk <= M * (1 - x[i,j_prime,k_prime]) + d[i,j_prime,k_prime] ), name=f"precedence_operations_inactive_{i}_{j}_{j_prime}_{k}")


        # constraint : if collab is not possible then z_auxilary[i,j,2] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                index_j = self.instance.jobs_struct[i][j]
                if self.instance.tasks_times[index_j][2] == -1 :
                    m.addConstr((z_auxilary[i,j,2] == 0), name=f"no_collab_{i}_{j}")


        # constraint : if two worker k1 and k2 are assigned to the same task (i,j) then the starting time of the task for both workers must be the same : d[i,j,k1] = d[i,j,k2]
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k1 in range(self.instance.nb_workers):
                    for k2 in range(k1+1, self.instance.nb_workers): # k2 > k1 to avoid duplicate constraints
                        m.addConstr((d[i,j,k1] <= d[i,j,k2] + M * (2 - (x[i,j,k1] + x[i,j,k2]))), name=f"same_start_time1_{i}_{j}_{k1}_{k2}")
                        m.addConstr((d[i,j,k2] <= d[i,j,k1] + M * (2 - (x[i,j,k1] + x[i,j,k2]))), name=f"same_start_time2_{i}_{j}_{k1}_{k2}")

    def _worker_of_the_first_operation_must_do_all_operations_of_the_job(self, m, x):

        # HYPOTHESE :
        # La première opération de chaque job doit être réalisé par un seul worker.
        # Si ce n'est pas possible alors ajouter une tache fictive au début du job pour l'affecter au worker qui sera assigné à toutes les opération du job en question

        # constraint :
        # For the first operation of each job at most 1 worker for we know it is the worker assigned to the all job
        for i in range(self.instance.nb_jobs):
            m.addConstr((gp.quicksum(x[i, 0, k] for k in range(self.instance.nb_workers)) <= 1), name=f"max_first_sub_operation_assignment_{i}")


        # constraint : if worker start the first task of a job then this worker must do all the tasks for this job
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    # print("ici->", i, j, k)
                    m.addConstr((x[i,0,k] <= x[i,j,k]), name=f"same_worker_operation_{i}_{j}_{k}")

    def _at_least_one_worker_with_level_greater_than_difficulty_of_task(self, m, x): # -> les taches en apprentissage doivent etre fait avec une personne de niveau ????

        # constraint : level of worker k must be >= difficulty of the operation assigned to worker k
        #              if 2 workers are asssigned to the same operation, at least one of the two workers must have a level higher than the difficulty of the opeation
        M = 2
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j] 
                    if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:
                        m.addConstr((x[i,j,k] + gp.quicksum(x[i,j,k_prime] for k_prime in range(self.instance.nb_workers) if k_prime != k) >= M*(x[i,j,k] - 1) + 2), name=f"at_most_two_workers_{i}_{j}_{k}") # at most 2 workers can be assigned to the same task if k do the task and he dont have the levels for it
                        
                        # La contrainte suivante permet de modéliser que si cette tache est fait par x_ijs et n'a pas le niveau alors l'autre personne avec lui doit l'avoir
                        m.addConstr((gp.quicksum(x[i,j,k_prime] * self.instance.levels_workers[k_prime][index_m] for k_prime in range(self.instance.nb_workers) if k_prime != k)) >= self.instance.tasks_difficulties[index_j] * x[i,j,k], name=f"at_least_one_worker_with_capacity_{i}_{j}_{k}") # if worker k do the sub op and he dont have the levels for it, at least one of the other workers assigned to the same sub-op must have the level for it

    # SOFT CONSTRAINTS
    def _at_least_one_worker_with_level_greater_than_difficulty_of_task_SOFT(self, m, x, penalty_levels): # -> SOFT
        # M = 2 
        # M = 100
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         for k in range(self.instance.nb_workers):
        #             index_j = self.instance.jobs_struct[i][j]
        #             index_m = self.instance.task_to_m[index_j] 
        #             if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:
        #                 m.addConstr((x[i,j,k] + gp.quicksum(x[i,j,k_prime] for k_prime in range(self.instance.nb_workers) if k_prime != k) >= M*(x[i,j,k] - 1) + 2 - penalty_level * self.instance.levels_workers[k][index_m]), name=f"at_most_two_workers_{i}_{j}_{k}") 
                        
        #                 M = 16
        #                 M = 100
        #                 print("iciiiiiiiiiii -> ", i, j, k)
        #                 m.addConstr((gp.quicksum(x[i,j,k_prime] * self.instance.levels_workers[k_prime][index_m] for k_prime in range(self.instance.nb_workers) if k_prime != k)) >= self.instance.tasks_difficulties[index_j] * x[i,j,k] - M * penalty_level, name=f"at_least_one_worker_with_capacity_{i}_{j}_{k}")

        pas_le_niveau = []
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                pas_le_niveau = []
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j] 
                    if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:
                        pas_le_niveau.append(x[i,j,k])

                # ne peuvent pas etre ensemble sur l'opération,
                if len(pas_le_niveau) > 0:
                    m.addConstr((gp.quicksum(pas_le_niveau) <= 1), name=f"at_most_one_worker_without_level_{i}_{j}") 
                        
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j] 
                    if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:
                        # peut faire la tache seul mais pénalité
                        M = 4
                        # si fait en apprentissage alors deux personnes et pas de penalité, si fait seul et pas le niveau le une penalité
                        m.addConstr((self.instance.levels_workers[k][index_m] * x[i,j,k] >= self.instance.tasks_difficulties[index_j] * x[i,j,k] - penalty_levels[i,j] + M *(1 - gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers)))), name=f"penalty_if_worker_without_level_do_task_{i}_{j}_{k}")

    # NE PRENS PAS ENCORE EN COMPTE TACHE SOLO SANS COMPETENCE (A FAIRE)
    def _teaching_effect_constraints(self, m, x, l):

        # constraint teaching effect
        # level of worker k for a metier after teaching effect can not be more than 1 unit higher than the initial level of worker k for this metier
        for k in range(self.instance.nb_workers):
            for metier in range(self.instance.nb_professions): # We fix a worker and a metier
                
                all_task_m_learning = 0 # number of times worker k do the task in metier
                for i in range(self.instance.nb_jobs):
                    for j in range(len(self.instance.jobs_struct[i])):
                        index_j = self.instance.jobs_struct[i][j]
                        
                        if self.instance.task_to_m[index_j] == metier and self.instance.levels_workers[k][metier] < self.instance.tasks_difficulties[index_j]: # Si sous-op s du metier metier assigné aux worker et qu'il n'avait pas le niveau -> apprentissage
                            all_task_m_learning += x[i,j,k] # number of times worker k do the sub_op in metier
 
                m.addConstr((l[k,metier] <= self.instance.levels_workers[k][metier] + all_task_m_learning*COEF_LEARNING), name=f"learning_effect_w{k}_metier{metier}") # learning effect for metier metier
                # contrainte suivante peut etre omis ?
                m.addConstr((l[k,metier] <= self.instance.levels_workers[k][metier] + 1), name=f"max_learning_effect_w{k}_metier{metier}") # max learning effect for metier 
                m.addConstr((l[k,metier] >= self.instance.levels_workers[k][metier]), name=f"min_level_metier{metier}") # level of worker k for metier metier can not be less than the initial level of worker k for metier 
                m.addConstr((l[k,metier] <= LEVEL_MAX), name=f"max_level_metier{metier}") # level of worker k for metier metier can not be more than 4 because the max difficulty of sub-op is 4
                        
    # # considérer phénomène d'oublie pour les corps de métier qu'un opérateur n'a pas éxectuer depuis un certains temps
    # def _forgetting_effect_constraints(self, m, x, forgetting):
    #     # forgetting[k][m] = 0 : pas de phénomène d'oublie
    #     # foretting[k][m] = 1 : phénomène d'oublie total 

    #     tab_do_task = [ [0 for metier in range(self.instance.nb_professions) ] for k in range(self.instance.nb_workers)] # tab_do_task[k][m] = nombre de tache du metier m fait par le worker k
    #     # print("tab_do_task ->", tab_do_task)
    #     for k in range(self.instance.nb_workers):  
    #         for metier in range(self.instance.nb_professions): # We fix a worker and a metier
                
    #             for i in range(self.instance.nb_jobs):
    #                 for j in range(len(self.instance.jobs_struct[i])):
                        
    #                     index_j = self.instance.jobs_struct[i][j]
    #                     if self.instance.task_to_m[index_j] == metier :
    #                         tab_do_task[k][metier] += x[i,j,k] # nombre de tache du metier index_m fait par le worker k
                        
                
    #             m.addConstr((forgetting[k, metier] >= 0), name=f"forgetting_positive_w{k}_m{metier}") # positive value of forgetting
    #             m.addConstr((forgetting[k, metier] >= self.instance.forgetting[k][metier] * (1 - tab_do_task[k][metier]) + (1 - tab_do_task[k][metier]) * COEF_FORGETTING), name=f"forgetting_effect_w{k}_m{metier}_if_w_{k}_do_not_do_task_m{metier}") # if k not do metier m so forgetting value can not be 0



    def _cognitive_load_constraints(self, m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, tab_count_tasks_has_tutor, tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, tab_count_tasks_with_collab, cognitive_load_total):

        ## TEACHING
        # contrainte pour savoir si un worker k à le niveau de compétence requis pour faire la tache O_ij
        M = LEVEL_MAX
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    m.addConstr((self.instance.levels_workers[k][index_m] + M * (1 - has_level[i,j,k]) >= self.instance.tasks_difficulties[index_j]), name=f"definition_theta_{i}_{j}_{k}") # if worker k has the level required -> tetha_ijsk = 0 or 1, if worker k doesn't have the level required -> tetha_ijsk = 0
                    m.addConstr((self.instance.levels_workers[k][index_m] - M * has_level[i,j,k] <= self.instance.tasks_difficulties[index_j] - EPS), name=f"definition_theta2_{i}_{j}_{k}") # if worker k has the level required -> tetha_ijsk = 1, if worker k doesn't have the level required -> tetha_ijsk = 0 or 1
                

        # contrainte pour savoir si k à fait la tache O_ij en tant que TUTEUR
        # linéarisation du ET LOGIQUE
        # is_tutor_ijk = x_ijk AND z_ij1 AND has_level_ijk
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    m.addConstr((is_tutor[i,j,k] <= x[i,j,k]), name=f"definition_tutor_{i}_{j}_{k}")
                    m.addConstr((is_tutor[i,j,k] <= z_auxilary[i,j,1]), name=f"definition_tutor2_{i}_{j}_{k}")
                    m.addConstr((is_tutor[i,j,k] <= has_level[i,j,k]), name=f"definition_tutor3_{i}_{j}_{k}")
                    m.addConstr((is_tutor[i,j,k] >= x[i,j,k] + z_auxilary[i,j,1] + has_level[i,j,k] - 2), name=f"definition_tutor4_{i}_{j}_{k}")
                    tab_count_tasks_has_tutor[k][index_m] += is_tutor[i,j,k] # nombre de taches en tant que tuteur de metier m pour le worker k

        # contrainte pour savoir si k à fait la tache O_ij en tant qu'APPRENTI
        # is_apprenti_ijk = x_ijk AND z_ij1 AND (1 - has_level_ijk)
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    m.addConstr((is_apprenti[i,j,k] <= x[i,j,k]), name=f"definition_app_{i}_{j}_{k}")
                    m.addConstr((is_apprenti[i,j,k] <= z_auxilary[i,j,1]), name=f"definition_app2_{i}_{j}_{k}")
                    m.addConstr((is_apprenti[i,j,k] <= 1 - has_level[i,j,k]), name=f"definition_app3_{i}_{j}_{k}")
                    m.addConstr((is_apprenti[i,j,k] >= x[i,j,k] + z_auxilary[i,j,1] - has_level[i,j,k] - 1), name=f"definition_app4_{i}_{j}_{k}")
                    tab_count_tasks_has_apprenti[k][index_m] += is_apprenti[i,j,k] # nombre de taches den tant que apprenti de metier m pour le worker k


        # contrainte pour calculer la charge cognitive des tuteurs lorsqu'il apprennent une tache à un apprenti
        m.addConstrs((cognitive_load_tutors[k,metier] == tab_count_tasks_has_tutor[k][metier] * COEF_TUTOR for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="count_tutor_tasks")

        # contrainte pour calculer la charge cognitive des apprentis lorsqu'ils apprennent une tache avec un tuteur
        m.addConstrs((cognitive_load_apprentis[k,metier] == tab_count_tasks_has_apprenti[k][metier] * COEF_APPRENTI for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="count_apprenti_tasks")



        ## COLLAB
        ## normalement pas besoin de verif s'ils ont bien le niveau
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    # Linéarisation du ET LOGIQUE pour savoir si la tache O_ijs est fait en collab par le worker k
                    # is_collab_ijsk = x_ijsk AND z_ijs2 : Si k à fait la tache et que cette tache est fait en collab
                    m.addConstr((is_collab[i,j,k] <= x[i,j,k]), name=f"definition_collab_{i}_{j}_{k}")
                    m.addConstr((is_collab[i,j,k] <= z_auxilary[i,j,2]), name=f"definition_collab2_{i}_{j}_{k}")
                    m.addConstr((is_collab[i,j,k] >= x[i,j,k] + z_auxilary[i,j,2] - 1), name=f"definition_collab3_{i}_{j}_{k}")
                    tab_count_tasks_with_collab[k][index_m] += is_collab[i,j,k]

        m.addConstrs((cognitive_load_collaboration[k,metier] == tab_count_tasks_with_collab[k][metier] * COEF_COLLAB for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="count_collaboration_tasks")

        ###### SUM OF COGNITIVE LOADS ######
        # J'ai ajouté nouveau tableau de variable, mais pourrait etre fait dans la fonction objectif directement en sommant les trois charges cognitives !!
        # A voir ce qui est plus simple pour la résolution du modèle
        m.addConstrs((cognitive_load_total[k,metier] == cognitive_load_tutors[k,metier] + cognitive_load_collaboration[k,metier] + cognitive_load_apprentis[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="cognitive_load_total")

    def _no_teaching_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,1] == 0), name=f"no_learning_tasks_{i}_{j}_{k}")

    def _no_collaboration_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,2] == 0), name=f"no_collaboration_tasks_{i}_{j}_{k}")

    def _no_solo_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,0] == 0), name=f"no_solo_tasks_{i}_{j}_{k}")

    def _all_collaboration_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,2] == 1), name=f"all_collaboration_tasks_{i}_{j}_{k}")

    def _all_solo_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,0] == 1), name=f"all_solo_tasks_{i}_{j}_{k}")

    def _constrained_makespan(self, m, C_max, penalty_makespan):
        # if makespan is grater than a certain thresold, we consider penalty for the exceeded part
        m.addConstr((C_max <= BORN_SUP_MAKESPAN + penalty_makespan), name=f"constrained_makespan")

    # NE PRENS PAS ENCORE EN COMPTE TACHE SOLO SANS COMPETENCE (A FAIRE)
    def _deadline_constraints_operation(self, m, x, d, z_auxilary, in_time, penalty_deadline, C, type="job"):
        """
        type : "operation" ou "job" pour savoir si on applique la contrainte sur les opérations ou sur les jobs
        """
        ########### Une tache (considérer jobs ou opérations ou sous-opérations) qui commence avant la date limite BORN_SUP_MAKESPAN doit se terminer avant celle ci
        ## -> forcer cette contrainte en dure pour le moment mais voir si on peut autoriser mais grande pénalité si on la dépasse pour ne pas rendre le modèle infaisable
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    
                    # in_time[i,j,k] = 1 alors d[i,j,k] <= BORN_SUP_MAKESPAN, 0 sinon
                    # Si k ne fait pas la tache O_ijs -> in_time[i,j,k] = 1
                    m.addConstr((d[i,j,k] >= BORN_SUP_MAKESPAN - M * in_time[i,j,k]), name=f"start_time_before_deadline_{i}_{j}_{k}")
                    m.addConstr((d[i,j,k] <= BORN_SUP_MAKESPAN + M * (1 - in_time[i,j,k])), name=f"start_time_before_deadline2_{i}_{j}_{k}")

                    if type == "operation":
                        index_j = self.instance.jobs_struct[i][j] # A VOIR -> Besoin pour calculer la fin de la tache mais avec variable f_ijk directement quand je l'aurais terminer les contraintes sur cette varibale f_ijk
                        # si k fait la tache : f_ijsk vrai fin
                            # si in_time = 1 -> f_ijsk doit se terminer avant OK
                            # si in_time = 0 -> f_ijsk non contraint par cette coontrainte OK
                        # si k ne fait pas la tache : f_ijsk = - M <= BORN_SUP_MAKESPAN donc OK

                        f_ijsk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] - M * (1 - x[i,j,k])
                        m.addConstr((f_ijsk <= BORN_SUP_MAKESPAN + penalty_deadline[i,j] + M * (1 - in_time[i,j,k])), name=f"end_time_before_deadline_{i}_{j}_{k}")

                    if type == "job":
                        m.addConstr((C[i] <= BORN_SUP_MAKESPAN + gp.quicksum(penalty_deadline[i,j] * in_time[i,j,k] for j in range(len(self.instance.jobs_struct[i])))), name=f"job_completion_before_deadline_{i}_{j}_{k}")


        self.PENALTY_DEADLINE = True

    def _hard_constraints_level_must_be_higher_if_solo(self, m, z_auxilary, ):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((z_auxilary[i,j,3] == 0), name=f"no_solo_without_level_{i}_{j}")

    def _build_indexes(self):
        self.indexes = {}
        # Useful for use uniquely the existing tuples of indices for the variables
        # Allow to reduce the number of variables and constraints
        assignment = [] # for varibles x, d, f, Delta_min
        operation = [] # for variable Level_min
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                operation.append((i, j))
                for k in range(self.instance.nb_workers):
                        assignment.append((i, j, k))
        self.indexes["assignment"] = assignment
        self.indexes["operation"] = operation

        sequencing = []
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for h in range(self.instance.nb_jobs):
                    for g in range(len(self.instance.jobs_struct[h])):
                        for k in range(self.instance.nb_workers):
                            sequencing.append((i, j, h, g, k))
        self.indexes["sequencing"] = sequencing

        mode = []
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                mode.append((i, j, 0)) # 1 si O_ij est fait en solo et que worker a le niveau
                mode.append((i, j, 1)) # 2 si O_ij est fait en apprentissage
                mode.append((i, j, 2)) # 3 si O_ij est fait en collab
                mode.append((i, j, 3)) # 4 si O_ij est fait en solo et que worker a pas le niveau
        self.indexes["mode"] = mode
        
    def _build_variables(self, m):

        x = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="x") # x[i, j, k] = 1 if operation j of job i is assigned to worker k
        d = m.addVars(self.indexes["assignment"], vtype=GRB.CONTINUOUS, name="d") # d[i, j, k] = starting time of operation j of job i if assigned to worker k
        C = m.addVars(self.instance.nb_jobs, vtype=GRB.CONTINUOUS, name="C") # C[i] = completion time of job i if minimize sum of C[i], completion time of all jobs if minimize C_max
        C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max") # C_max = makespan
        delta = m.addVars(self.indexes["sequencing"], vtype=GRB.BINARY, name="delta") # delta[i, j, h, g, k] = 1
        # y = m.addVars(self.instance.nb_workers,self.instance.nb_sub_operations, vtype=GRB.INTEGER, name="y") # y[k] = number of sub-operations assigned to worker k
        
        # learning AND forgetting effects variables
        l = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="l") # l[k,m] = level of worker k before performing metier m after run of the PL
        # forgetting = m.addVars(self.instance.nb_workers, self.instance.nb_professions, lb=0, vtype=GRB.CONTINUOUS, name="forgetting") # forgetting[k,m] = niveau de forgetting pour le worker k et le metier m, utilisé pour modéliser les effets d'oublie
        
        f = m.addVars(self.indexes["assignment"], vtype=GRB.CONTINUOUS, name="f") # f[i,j,k] = completion time of operation j of job i if assigned to worker k -- Ajout de cette variable pour prendre en compte le fait que la duré d'une tache peut etre different selon si fait en solo, en apprentissage ou en collab
        z_auxilary = m.addVars(self.indexes["mode"], vtype=GRB.INTEGER, name="z_auxilary") # z[i,j,0] = 1 if O_ij is done in solo, z[i,j,1] = 1 if O_ij is done in apprentissage

        # Linearisation min pour savoir si une tache est fait en apprentissage ou en collab
        Level_min = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS, name="Level_min") #vaut le level min d'un worker sur O_ij
        Delta_min = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="Delta_min") # pour linearisation du min


        # Pour la partie Ergonomie (Faire augmenter le niveau de fatigue cognitif des workers qui enseignent des taches pour lesquelles ils ont le niveau requis)
        
        ## tutor part
        is_tutor = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_tutor") # is_tutor[i,j,k] = 1 si O_ij est fait par k avec un apprenti
        has_level = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="has_level") # has_level[i,j,k] = 1 si k à le niveau de compétence requis pour faire O_ij
        cognitive_load_tutors = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="cognitive_load_tutors") # cognitive_load_tutors[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier avec un apprenti
        
        ## apprenti part
        is_apprenti = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_apprenti") # is_apprenti[i,j,k] = 1 si O_ij est fait par k en apprentissage
        cognitive_load_apprentis = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="cognitive_load_apprentis") # cognitive_load_apprentis[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier en apprentissage

        ## collaboration part
        is_collab = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_collab") # is_collab[i,j,k] = 1 si O_ij est fait par k en collaboration
        cognitive_load_collaboration = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="cognitive_load_collaboration") # cognitive_load_collaboration[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier en collaboration

        ## load tutor + load collab + load apprenti
        cognitive_load_total = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, name="cognitive_load_total") # somme des charge cognitive de tutorat et de collaboration pour chaque worker et chaque métier, utilisé pour l'objectif de minimisation de la charge cognitive
        

        ## Pénalité pour les taches qui dépassent la durée de la période courante - Si toutes les taches ne peuvent être réalisées dans la période considérée
        # si une tache dépasse la durée de la période courante, elle est considéré comme une pénalité dans le calcul du makespan et pourra être traité dans une autre période
        penalty_makespan = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="penalty_makespan") # pénalité pour les taches qui dépassent la durée de la période courante, utilisé pour le multi-période
        penalty_deadline = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS, lb=0, name="penalty_deadline") # pénalité pour les taches qui sont éxécuté pendant la date limite BORN_SUP_MAKESPAN

        # SOFT CONSTRAINTS OF LEVELS
        # penalty_levels = m.addVar(vtype=GRB.CONTINUOUS , lb=0, name="penalty_levels") # pénalité pour les taches fait par des workers qui n'ont pas le niveau requis pour les faire, utilisé pour la contrainte de niveau en soft (si le worker fait alors c'est seul pas avec un tuteur non compétent !!!!)
        penalty_levels = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS , lb=0, name="penalty_levels") # pénalité pour chaque tache fait en solo par un worker non compétent

        # Variable permettant de savoir si une tache à débuter avant la date limite BORN_SUP_MAKESPAN
        # Pour faire en sorte que cette tache doit se terminer avant la date limite BORN_SUP_MAKESPAN ou alors si elle dépasse cette date ajouté en pénalité
        in_time = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="in_time") # in_time[i,j,k] = 1 si O_ij commence avant la date limite BORN_SUP_MAKESPAN

        return x, d, C, C_max, delta, l, f, z_auxilary, Level_min, Delta_min, is_tutor, has_level, cognitive_load_tutors, is_apprenti, cognitive_load_apprentis, is_collab, cognitive_load_collaboration, cognitive_load_total, penalty_makespan, penalty_deadline, in_time, penalty_levels#, forgetting

    def _build_helper_variables(self):
        # Ce ne sont pas des variables de décision du modèle 
        # mais des variables pour aider à la construction du modèle et le calcul de certaines contraintes ou de l'objectif
        ######################### VARIBALES PROGRAMME #########################
   
        # variable du programme permettant de caluler le nombre de tache fait par k avec un apprenti pour les tache de métier m
        self.tab_count_tasks_has_tutor = [ [0 for m in range(self.instance.nb_professions)] for k in range(self.instance.nb_workers) ] 

        # variable du programme permettant de caluler le nombre de tache fait par k en collab pour les tache de métier m
        self.tab_count_tasks_with_collab = [ [0 for m in range(self.instance.nb_professions)] for k in range(self.instance.nb_workers) ] 

        # variable du programme permettant de caluler le nombre de tache fait en tant qu'apprenti par k pour les tache de métier m
        self.tab_count_tasks_has_apprenti = [ [0 for m in range(self.instance.nb_professions)] for k in range(self.instance.nb_workers) ]

        # Variable pour savoir si on a utilisé la pénalité de deadline dans les contraintes
        self.PENALTY_DEADLINE = False 
    
    def _build_model(self, objective, weight, priority, time_limit=None, constraints_config= None, verbose=False):
        """
        Construit le modèle de programmation linéaire
        
        Args:
            objective (int): l'objectif à optimiser, peut être "makespan", "skill", "both" ou "lexicographic" ou "cognitive_load_total"
            weight (list): les poids à accorder à chaque objectifs (makespan, skill) si objective = "both", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (makespan, skill) si objective = "lexicographic", n'est pas utilisé sinon
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            time_limit (int): la limite de temps pour l'optimisation

        Returns:
            m (gp.Model): le modèle de programmation linéaire construit
        """
        m = gp.Model(f"dual_resource_scheduling_objective_{objective}")

        # indexes
        self._build_indexes()

        # variables
        (
            x,
            d, 
            C, 
            C_max, 
            delta, 
            l, 
            f, 
            z_auxilary, 
            Level_min, 
            Delta_min, 
            is_tutor, 
            has_level, 
            cognitive_load_tutors, 
            is_apprenti, 
            cognitive_load_apprentis, 
            is_collab, 
            cognitive_load_collaboration, 
            cognitive_load_total, 
            penalty_makespan, 
            penalty_deadline, 
            in_time,
            penalty_levels
            # forgetting
        ) = self._build_variables(m)

        self._build_helper_variables()

        # constraints

        # voir si meilleure de mettre ces conditions dans une fonction _build_constraints
        self._add_physical_constraints(m, x, d, C, C_max, z_auxilary, Level_min, Delta_min, f, delta)
        self._teaching_effect_constraints(m, x, l)
        # self._forgetting_effect_constraints(m, x, forgetting)


        self._cognitive_load_constraints(m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, self.tab_count_tasks_has_tutor, self.tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, self.tab_count_tasks_with_collab, cognitive_load_total)
        # self._at_least_one_worker_with_level_greater_than_difficulty_of_task(m, x)
        self._at_least_one_worker_with_level_greater_than_difficulty_of_task_SOFT(m, x, penalty_levels)
        self._worker_of_the_first_operation_must_do_all_operations_of_the_job(m, x)
        
        if constraints_config is not None:
            
            if constraints_config.get("no_teaching_tasks", False) :
                self._no_teaching_tasks(m, z_auxilary)
            
            if constraints_config.get("no_collaboration_tasks", False) :
                self._no_collaboration_tasks(m, z_auxilary)
            
            if constraints_config.get("no_solo_tasks", False) :
                self._no_solo_tasks(m, z_auxilary)
            
            if constraints_config.get("all_collaboration_tasks", False) :
                self._all_collaboration_tasks(m, z_auxilary)
            
            if constraints_config.get("all_solo_tasks", False) :
                self._all_solo_tasks(m, z_auxilary)
            
            if constraints_config.get("constrained_makespan", False) :
                self._constrained_makespan(m, C_max, penalty_makespan)
            
            if constraints_config.get("deadline_constraints_operation", False) :
                self._deadline_constraints_operation(m, x, d, z_auxilary, in_time, penalty_deadline, C, type="job") # type = "operation" ou "job" pour savoir si on applique la contrainte sur les opérations ou sur les jobs
            
            if constraints_config.get("hard_constraints_level_must_be_higher_if_solo", False) :
                self._hard_constraints_level_must_be_higher_if_solo(m, z_auxilary)
        
########################################################################
########################### OBJECTIVE FUNCTION #########################
########################################################################
        if time_limit is not None:
            m.Params.TimeLimit = time_limit

        m.Params.SolFiles = "../results/intermediate_solutions.sol"

        if self.PENALTY_DEADLINE:
            penalty_deadline_obj = gp.quicksum(penalty_deadline[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
        else:
            penalty_deadline_obj = 0

        print("penalty deadline obj:", penalty_deadline_obj)

        # cognitive_load_tutors_obj = gp.quicksum(cognitive_load_tutors[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        # cognitive_load_collab_obj = gp.quicksum(cognitive_load_collaboration[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        
        cognitive_load_total_obj = gp.quicksum(cognitive_load_total[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        skill_obj = gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))

        cognitive_load_total_obj += penalty_deadline_obj
        skill_obj -= penalty_deadline_obj
        penalty_levels_obj = gp.quicksum(penalty_levels[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
        # forgetting_obj = gp.quicksum(forgetting[k, m] for k in range(self.instance.nb_workers) for m in range(self.instance.nb_professions))


        if objective == "makespan":
            m.setObjectiveN(penalty_levels_obj, index=0, priority=2, name="penalty_levels_obj")
            m.setObjectiveN(C_max, index=1, priority=1, name="makespan_obj")
            m.modelSense = GRB.MINIMIZE

            # m.setObjective(C_max + penalty_levels_obj * 10000, GRB.MINIMIZE)
            # m.setObjective(penalty_makespan + penalty_deadline_obj, GRB.MINIMIZE) # in minimizing the penalty for the tasks that exceed the current period, we minimize the makespan
        
        elif objective == "cognitive_load_total":
            m.setObjective(cognitive_load_total_obj + penalty_levels_obj *1000, GRB.MINIMIZE)

        # lorsque l'on optimise le skill, une fois optimisé on cherche à minimiser le makespan pour ne pas avoir de soultions abbérantes
        elif objective == "skill": 
            m.setObjectiveN(penalty_levels_obj, index=0, priority=2, name="penalty_levels_obj")
            m.setObjectiveN(- skill_obj, index=1, priority=1, name="maximize_skill_levels")
            m.setObjectiveN(C_max, index=2, priority=0, name="minimize_makespan")
            m.modelSense = GRB.MINIMIZE


        # en priorité minimiser la difference entre le niveau du worker et les difficulté des taches qu'il fait seul : penalty_levels_obj
        elif objective == "lexicographic":
            m.setObjectiveN(penalty_levels_obj, index=0, priority=4, name="penalty_levels_obj")
            m.setObjectiveN(C_max , index=1, priority=priority[0]+1, name="minimize_makespan_obj")
            m.setObjectiveN(-skill_obj, index=2, priority=priority[1]+1, name="minimize_minus_skill_levels_obj")
            m.setObjectiveN(cognitive_load_total_obj, index=3, priority=priority[2]+1, name="minimize_cognitive_load_total_obj")
            # m.setObjectiveN(forgetting_obj, index=4, priority=0, name="minimize_forgetting_obj")
            m.modelSense = GRB.MINIMIZE

        elif objective == "three":
            m.setObjectiveN(C_max, index=0, weight=weight[0], name="minimize_makespan")
            m.setObjectiveN(-skill_obj, index=1, weight=weight[1], name="minimize_minus_skill_levels")
            m.setObjectiveN(cognitive_load_total_obj, index=2, weight=weight[2], name="minimize_cognitive_load_total")
            m.modelSense = GRB.MINIMIZE

        else:
            raise ValueError("objective doit être 'makespan', 'skill', 'three', 'lexicographic' ou 'cognitive_load_tutors'")
        


        # sum_Ci_obj = gp.quicksum(C[i] for i in range(self.instance.nb_jobs))

        # # minimsier somme des complétudes des jobs
        # m.setObjective(sum_Ci_obj, GRB.MINIMIZE)

        # # double objectif : minimiser le makespan et maximiser le niveau de compétence des travailleurs
        # m.setObjective(0.5 * C_max - 0.5 * skill_obj, GRB.MINIMIZE)

        m.write(f"../results/model_{objective}.lp")
        return m

    def solve(self, objective="makespan", weight=[0,0], priority=[0,1], time_limit=None, constraints_config=None, verbose=False):
        """
        Résout le modèle et affiche les résultats
        
        Args:
            objective (str): l'objectif à optimiser, peut être "makespan", "skill", "both" ou "lexicographic"
            weight (list): les poids à accorder à chaque objectifs (makespan, skill, cognitive_load_total) si objective = "both", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (makespan, skill, cognitive_load_total) si objective = "lexicographic", n'est pas utilisé sinon
            time_limit (int): la limite de temps pour l'optimisation
            constraints_config (dict): la configuration des contraintes à ajouter au modèle, par exemple {"no_teaching_tasks": True, "constrained_makespan": True, ...}
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            
        Returns:
            (Solution): une instance de la classe Solution contenant les résultats de la résolution du modèle
        """

        assert objective in ["makespan", "skill", "three", "lexicographic", "cognitive_load_total"], "objective doit être 'makespan', 'skill', 'three', 'lexicographic' ou 'cognitive_load_total'"
        assert len(weight) == 3, "weight doit être une liste de trois éléments"
        assert len(priority) == 3, "priority doit être une liste de trois éléments"

        m = self._build_model(objective, weight, priority, time_limit, constraints_config, verbose)

        if verbose == False:
            m.setParam('OutputFlag', 0) # to disable gurobi output

        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found with objective value:", m.objVal)
            m.write("../results/solution.sol")
            self.write_objectives_values(m, m.NumObj, "../results/objectives_values.txt")

            
        else:
            print("No optimal solution found. Status code:", m.status)
            return

        res = [] # liste de tuples (nom_variable, valeur_variable) pour les variables du modèle dans la solution optimale
        
        # Query number of multiple objectives, and number of solutions
        nSolutions = m.SolCount
        nObjectives = m.NumObj
        print("nObjectives", nObjectives)
        print("nSolutions", nSolutions)
        if nObjectives > 1:
            for o in range(nObjectives): # On récupère la valeur de chaque objectif pour la solution optimale
                m.params.ObjNumber = o
                # print("m.ObjNVal", m.ObjNVal)
                res.append(('Obj'+str(o), m.ObjNVal))
        
        if verbose:
            print("Problem has", nObjectives, "objectives")
            print("Gurobi found ", nSolutions, "solutions")
            print("***********************")
            
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
        for name, value in zip(names, values): # variables du modèle avec leur valeur dans la solution optimale
            res.append((name, value))

        
        if verbose :
            print("objective value:", m.objVal)
            print(res)
        return Solution(res, self.instance)    

class Solution:
    def __init__(self, var_list, instance):
        # Les matrices suivantes possèdent beaucoup de zéros car elles sont de la taille maximale.
        self.x = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers)) # x[i, j, k] = 1 if operation j of job i is assigned to worker k, 0 otherwise
        self.d = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers))
        self.C = np.zeros(instance.nb_jobs)
        self.C_max = 0
        self.delta = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_jobs, instance.max_nb_operations, instance.nb_workers))
        
        self.l = np.zeros((instance.nb_workers, instance.nb_professions))
        # self.forgetting = np.zeros((instance.nb_workers, instance.nb_professions))
        
        
        self.f = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers))
        # prise en compte qua tache peut etre fait seul sans level 
        self.z_auxilary = np.zeros((instance.nb_jobs, instance.max_nb_operations, 4)) # z_auxilary[i,j,mode] = 1 if operation j of job i is done en solo (mode=0) ou en apprentissage (mode=1) ou en collab (mode=2)
        
        # ergonomic variables
        self.is_tutor = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers)) # is_tutor[i,j,k] = 1 if worker k is tutor for operation j of job i, 0 otherwise
        self.cognitive_load_tutors = np.zeros((instance.nb_workers, instance.nb_professions)) # cognitive_load_tutors[k, m] = charge cognitive pour le worker k liée à l'apprentissage  en tant que tuteur pour le métier m
        self.cognitive_load_apprentis = np.zeros((instance.nb_workers, instance.nb_professions)) # cognitive_load_apprentis[k, m] = charge cognitive pour le worker k liée à l'apprentissage  en tant que apprenti pour le métier m
        self.cognitive_load_collaboration = np.zeros((instance.nb_workers, instance.nb_professions)) # cognitive_load_collaboration[k, m] = charge cognitive pour le worker k liée à la collaboration pour le métier m
        self.cognitive_load_total = np.zeros((instance.nb_workers, instance.nb_professions)) # cognitive_load_total[k, m] = charge cognitive totale pour le worker k pour le métier m
        
        # penalty for soft CONSTRAINTS :
        self.penalty_levels = np.zeros((instance.nb_jobs, instance.max_nb_operations)) # penalty_levels[i,j] = pénalité pour l'opération j du job i si elle est faite en solo par un worker qui n'a pas le niveau requis pour la faire

        # objective values
        self.objective_values = {}

        # print("var_list", var_list)
        for v in var_list:

            if v[0][0][0] == "x":
                indices = v[0][2:-1].split(",") # x[i, j, k] -> indices = [i, j, k]
                i, j, k = int(indices[0]), int(indices[1]), int(indices[2])
                # print(f"x[{i}, {j}, {k}] = {v[1]}")
                self.x[i, j, k] = v[1]

            elif v[0][0] == "d" and v[0][1] == "[" : # == "[" pour éviter confusion avec variable delta
                indices = v[0][2:-1].split(",") # d[i, j, k] -> indices = [i, j, k]
                i, j, k = int(indices[0]), int(indices[1]), int(indices[2])
                self.d[i, j, k] = v[1]

            elif v[0][0] == "C" and v[0][1] != "_": # C[i] -> indices = [i]
                indices = v[0][2:-1].split(",")
                i = int(indices[0])
                self.C[i] = v[1]

            elif v[0] == "C_max":
                self.C_max = v[1]

            elif v[0][0] == "d" and v[0][1] == "e" : # delta[i, j, h, g, k] -> indices = [i, j, h, g, k]
                indices = v[0][6:-1].split(",")
                i, j, h, g, k = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3]), int(indices[4])
                self.delta[i, j, h, g, k] = v[1]

            elif v[0][0] == "l" : # l[k, m] -> indices = [k, m]
                indices = v[0][2:-1].split(",")
                k, m = int(indices[0]), int(indices[1])
                self.l[k, m] = v[1]

            elif v[0][:2] == "f[": # f[i, j, k] -> indices = [i, j, k]
                indices = v[0][2:-1].split(",")
                i, j, k = int(indices[0]), int(indices[1]), int(indices[2])
                self.f[i, j, k] = v[1]

            elif v[0][:10] == "z_auxilary": # z_auxilary[i, j, z] -> indices = [i, j, z]
                indices = v[0][11:-1].split(",")
                i, j, z = int(indices[0]), int(indices[1]), int(indices[2])
                self.z_auxilary[i, j, z] = v[1]

            elif v[0][:8] == "is_tutor": # is_tutor[i, j, k] -> indices = [i, j, k]
                indices = v[0][9:-1].split(",")
                i, j, k = int(indices[0]), int(indices[1]), int(indices[2])
                # print(f"is_tutor[{i}, {j}, {k}] = {v[1]}")
                self.is_tutor[i, j, k] = v[1]

            elif v[0][:21] == "cognitive_load_tutors" :
                indices = v[0][22:-1].split(",")
                k, metier = int(indices[0]), int(indices[1])
                # print(f"cognitive_load_tutors[{k}, {metier}] = {v[1]}")
                self.cognitive_load_tutors[k, metier] = v[1]
            
            elif v[0][:28] == "cognitive_load_collaboration" :
                indices = v[0][29:-1].split(",")
                k, metier = int(indices[0]), int(indices[1])
                # print(f"cognitive_load_collaboration[{k}, {metier}] = {v[1]}")
                self.cognitive_load_collaboration[k, metier] = v[1]

            elif v[0][:24] == "cognitive_load_apprentis" :
                indices = v[0][25:-1].split(",")
                k, metier = int(indices[0]), int(indices[1])
                # print(f"cognitive_load_apprentis[{k}, {metier}] = {v[1]}")
                self.cognitive_load_apprentis[k, metier] = v[1]

            elif v[0][:20] == "cognitive_load_total" :
                indices = v[0][21:-1].split(",")
                k, metier = int(indices[0]), int(indices[1])
                # print(f"cognitive_load_total[{k}, {metier}] = {v[1]}")
                self.cognitive_load_total[k, metier] = v[1]

            elif v[0][:3] == "Obj":
                index_obj = int(v[0][3:])
                self.objective_values[index_obj] = v[1]

            elif v[0][:14] == "penalty_levels":
                indices = v[0][15:-1].split(",")
                i, j = int(indices[0]), int(indices[1])
                # print(f"penalty_levels[{i}, {j}] = {v[1]}")
                self.penalty_levels[i, j] = v[1]

            # elif v[0][:10] == "forgetting":
            #     indices = v[0][11:-1].split(",")
            #     k, m = int(indices[0]), int(indices[1])
            #     self.forgetting[k, m] = v[1]

    # fonction __str__ pas à jours 
    def __str__(self):
        res = (f"x: {self.x.shape} \n{self.x}\n"
               f"d: {self.d.shape} \n{self.d}\n"
               f"C: {self.C.shape} \n{self.C}\n"
               f"C_max: {self.C_max}\n"
            #    f"delta: {self.delta.shape}\n{self.delta}\n"
               f"l: {self.l.shape}\n{self.l}\n")
        
        return res


if __name__ == "__main__":
    
    # res = read_file("../data/data_temp.test")
    res = read_file("../data/data_temp.test")
    
    instance = Instance()
    instance.from_dictionary(res)

    print(instance)
    model = Model(instance)
    
    # # s = model.solve(objective="lexicographic", weight=[0.5, 0.5, 0.5], priority=[2, 1, 0], verbose=True)
    # s = model.solve(objective="makespan", weight=[0.5, 0.5, 0.5], priority=[2, 1, 0], verbose=True)
    s = model.solve(objective="lexicographic", weight=[0, 0, 0], priority=[2, 1, 0], verbose=True)
    resume_levels_workers(s, instance)
    # print(s)

    gantt_chart(s, instance, color=3, verbose=True)
    # # plot_levels_workers(s, instance, verbose=True)



    # instance_generated = Instance()
    # instance_generated.from_random(seed=42)
    # print(instance_generated)
    # # res = read_file("../data/data_temp_2.test")
    # # instance_generated = Instance()
    # # instance_generated.from_dictionary(res)
    # # print(instance_generated)

    # # soft constraint on the level for solo tasks : it can be lower but with a penalty in the objective function
    # model_soft = Model(instance_generated)
    # s_soft = model_soft.solve(objective="makespan", weight=[0.5, 0.5, 0.5], priority=[2, 1, 0], verbose=True)
    # gantt_chart(s_soft, instance_generated, color=3, verbose=True)

    # # # hard constraint of level must be higher if solo task
    # # model_hard = Model(instance_generated)
    # # s_hard = model_hard.solve(objective="makespan", constraints_config={"hard_constraints_level_must_be_higher_if_solo": True}, verbose=True)
