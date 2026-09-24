import gurobipy as gp
from gurobipy import GRB
from utils import *
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from Instance import *
from Solution import *
from Constante import *





class Model:

    # Séparer modele et solve 
    def __init__(self, instance):
        self.instance = instance
        self.BORNE_SUP_MAKESPAN = None

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
            # print("ici -> ", nObjectives)
            if nObjectives == 1:
                f.write(f"Obj: {m.ObjVal}\n")
            else: 
                for o in range(nObjectives):
                    m.params.ObjNumber = o
                    f.write(f"Obj{o}: {m.ObjNVal}\n")
        f.close()

    def _add_physical_constraints(self, m, x, d, C, C_max, z_auxilary, Level_min, Delta_min, f, delta, job_done, task_done):
    
        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint :
        # At most 2 workers per tasks
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])): # for all operations except the first one of each job
                m.addConstr((gp.quicksum(x[i, j, k] for k in range(self.instance.nb_workers)) <= 2), name=f"max_assignment_operation_{i}_{j}")
                
        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint : 
        # Makespan >= f[i,j,k] for all i, j, k
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((C_max >= f[i,j,k]), name=f"makespan_{i}_{j}_{k}")
    

        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint : 
        # auxilary variables z_ij^(solo), z_ij^(apprentissage), z_ij^(collab) and z_ij^(solo_no_level) do not take value of 1 for the same operation
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])): 

                # chaque opération est soit en solo, en apprentissage, collab ou solo no level
                m.addConstr((z_auxilary[i,j,0] + z_auxilary[i,j,1] + z_auxilary[i,j,2] + z_auxilary[i,j,3] <= 1), name=f"z_assignment_1_{i}_{j}") # before it was == 1 but i consider all task are assinged, but not now, we can have tasks not assigned
                
                # z_ij fixed according to the number of workers assigned to the operation
                m.addConstr(((gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers)) == 1 * z_auxilary[i,j,0] + 1 * z_auxilary[i,j,3] + 2 * z_auxilary[i,j,1] + 2 * z_auxilary[i,j,2])), name=f"z_assignment_2_{i}_{j}")
                

        ##==================================================================================================================================
        ##==================================================================================================================================


        # Connaitre le level du worker le moins compétent associé à une operation O_ij
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
                
                # comme une tache peut ne pas être affécter alors passer de "== 1" à "<= 1"
                m.addConstr((gp.quicksum(Delta_min[i,j,k] for k in range(self.instance.nb_workers)) <= 1), name=f"linearization_min_binary_{i}_{j}") # Delta_min_ijsk doit prendre 1 pour le k tq il a le level minimal pour cette tache
        
                # Pour force Delta_min[ijk] à etre à 1 si l'operation a ete assigné à au moins 1 worker (AJOUTE RECEMMENT LE 12/06/2026) car apercu d'un beugue
                m.addConstr((gp.quicksum(Delta_min[i,j,k] for k in range(self.instance.nb_workers)) >= task_done[i,j]), name=f"linearization_min_binary3_{i}_{j}") # si la tache est faite alors il y a au moins un worker avec Delta_min_ijk = 1 pour cette tache
                

        
        # forcer que Delta_min[ijk] peut prendre la valeur de 1 que pour un worker affécté à la tache
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((Delta_min[i,j,k] <= x[i,j,k]), name=f"linearization_min_binary2_{i}_{j}_{k}") # Delta_min_ijsk doit être égal à 0 si le worker k n'est pas assigné à la tache (i,j)



        # constraint : to know the mode of an operation (solo, apprentissage or collab) with the variable Level_min to fixe auxilary variables z_ijs0, z_ijs1, z_ijs2
        M = LEVEL_MAX
        

        ########### NEW -> SOFT CONSTRAINTS with penalty in the objective function if not respected ##########
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
   
                    m.addConstr((self.instance.levels_workers[k][index_m] - self.instance.tasks_difficulties[index_j] * x[i, j, k] >=  - M * z_auxilary[i, j, 3] - M * z_auxilary[i, j, 1]), name=f"level_min_difficulty_beta0_{i}_{j}_{k}")
                    m.addConstr((self.instance.levels_workers[k][index_m] - (self.instance.tasks_difficulties[index_j] - EPS) * x[i,j,k] <=  (1 - x[i,j,k]) * M + (M * z_auxilary[i,j,0]) + (M * z_auxilary[i,j,1]) + (M * z_auxilary[i,j,2])), name=f"level_min_difficulty_beta1_{i}_{j}_{k}")


        ########## NEW -> SOFT CONSTRAINTS with penalty in the objective function if not respected ##########
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                index_j = self.instance.jobs_struct[i][j]
                nb_pers_ij = gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))

                m.addConstr((Level_min[i,j] + M * z_auxilary[i,j,1] + M * z_auxilary[i,j,3] >= self.instance.tasks_difficulties[index_j]), name=f"level_min_difficulty_beta2_{i}_{j}")
                m.addConstr((Level_min[i,j] - M * z_auxilary[i,j,2] <= self.instance.tasks_difficulties[index_j] - EPS + M*(2-nb_pers_ij)), name=f"level_min_difficulty_beta3_{i}_{j}")


        # constraints : 
        # -> s'il fait la tache f = d + times
        # si il fait pas f libre
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    time = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3] * PERC_SOLO_NO_LEVEL_TIME)
   
                    m.addConstr((f[i,j,k] == time), name=f"definition_f3_{i}_{j}_{k}") # si x[i,j,k] = 1 alors f[i,j,k] = d[i,j,k] + time, sinon f[i,j,k] est libre mais doit respecter les autres contraintes du modeles pour ne pas être trop grand

        ##==================================================================================================================================
        ##==================================================================================================================================
        
        # constraint : 
        # OVERLAP
        M = 1000
        for k in range(self.instance.nb_workers):
            for i in range(self.instance.nb_jobs):
                for j in range(len(self.instance.jobs_struct[i])):
                    for h in range(self.instance.nb_jobs):
                        for g in range(len(self.instance.jobs_struct[h])):
                            if (i, j) != (h, g):
                                index_j = self.instance.jobs_struct[i][j]
                                index_g = self.instance.jobs_struct[h][g]

                                # En mettant == j'ai status.code = 4 de Gurobi (non borné), en mettant >= j'ai status.code = 2 (optimal)
                                # donc le solveur force f[x1,x2,x3,x4] à être petit
                                m.addConstr(f[h,g,k] >= d[h,g,k] + self.instance.tasks_times[index_g][0] * z_auxilary[h,g,0] + self.instance.tasks_times[index_g][1] * z_auxilary[h,g,1] + self.instance.tasks_times[index_g][2] * z_auxilary[h,g,2] + (self.instance.tasks_times[index_g][0] * z_auxilary[h,g,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[h, g, k]))
                                m.addConstr(f[i,j,k] >= d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i, j, k]))
                                
                                f_hgk = d[h,g,k] + self.instance.tasks_times[index_g][0] * z_auxilary[h,g,0] + self.instance.tasks_times[index_g][1] * z_auxilary[h,g,1] + self.instance.tasks_times[index_g][2] * z_auxilary[h,g,2] + (self.instance.tasks_times[index_g][0] * z_auxilary[h,g,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[h, g, k])
                                f_ijk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i, j, k])
                                
                                m.addConstr((d[i,j,k] >= f_hgk - M * delta[i,j,h,g,k]), name=f"overlap1_{i}_{j}_{h}_{g}_{k}")
                                m.addConstr((d[h,g,k] >= f_ijk - M * (1 - delta[i,j,h,g,k])), name=f"overlap2_{i}_{j}_{h}_{g}_{k}")

        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint : 
        # Si Operation non affecté pour worker k alors d_ijk = 0
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((d[i,j,k] <= M * x[i,j,k]), name=f"start_time_zero_if_not_assigned_{i}_{j}_{k}")

        ##==================================================================================================================================
        ##==================================================================================================================================

       
        # constraint : 
        # PRECEDENCE of operation of the same job
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for j_prime in range(len(self.instance.jobs_struct[i])):
                    for k in range(self.instance.nb_workers):
                        if (j != j_prime) and (self.instance.constraints_precedence_operations[i,j,j_prime] == 1): # if O_ij must be performed before operation O_ij'
                            index_j = self.instance.jobs_struct[i][j]

                            f_ijk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] + (self.instance.tasks_times[index_j][0] * z_auxilary[i,j,3]*PERC_SOLO_NO_LEVEL_TIME) - M * (1 - x[i,j,k])
                            for k_prime in range(self.instance.nb_workers):
                                    m.addConstr((f_ijk <= M * (1 - x[i,j_prime,k_prime]) + d[i,j_prime,k_prime]  ), name=f"precedence_operations_inactive_{i}_{j}_{j_prime}_{k}_{k_prime}")

                            # ajout du fait que la tache i+1 doit etre effectué que si la tache i est effectué pour éviter les problèmes de tache non affecté et de precedence
                            # car maintenant des taches peuvent ne pas etre affecté 
                            m.addConstr((task_done[i,j] >= task_done[i,j_prime]), name=f"precedence_operations_task_done_{i}_{j}_{j_prime}")
            
        ##==================================================================================================================================
        ##==================================================================================================================================
     
        # constraint : 
        # Si 1 worker sur O_ij alors task_done_ij = 1
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((task_done[i,j] >= x[i,j,k]), name=f"precedence_operations_task_done2_{i}_{j}")


        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint : 
        # if collab is not possible then z_auxilary[i,j,2] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                index_j = self.instance.jobs_struct[i][j]
                if self.instance.tasks_times[index_j][2] == -1 :
                    m.addConstr((z_auxilary[i,j,2] == 0), name=f"no_collab_{i}_{j}")

        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint : 
        # if two worker k1 and k2 are assigned to the same task (i,j) then the starting time of the task for both workers must be the same : d[i,j,k1] = d[i,j,k2]
        M = 1000
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k1 in range(self.instance.nb_workers):
                    for k2 in range(k1+1, self.instance.nb_workers): # k2 > k1 to avoid duplicate constraints
                        m.addConstr((d[i,j,k1] <= d[i,j,k2] + M * (2 - (x[i,j,k1] + x[i,j,k2]))), name=f"same_start_time1_{i}_{j}_{k1}_{k2}")
                        m.addConstr((d[i,j,k2] <= d[i,j,k1] + M * (2 - (x[i,j,k1] + x[i,j,k2]))), name=f"same_start_time2_{i}_{j}_{k1}_{k2}")

        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint :
        # if task (i,j) is assigned to at least one worker then task_done[i,j] = 1, else task_done[i,j] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((task_done[i,j] <= gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))), name=f"task_done_definition_{i}_{j}")


        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint :
        # if all tasks of a job i are done then job_done[i] = 1, else job_done[i] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((job_done[i] <= task_done[i,j]), name=f"job_done_definition_{i}_{j}") # job_done[i] = 1 if all tasks of job i are done, else job_done[i] = 0
        

        ##==================================================================================================================================
        ##==================================================================================================================================

        # constraint :
        # Un opérateur ne peut faire une tache en tant qu'apprenti si son niveau de différence avec la tache est > LEVEL_DIFFERENCE
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    if self.instance.levels_workers[k][index_m] + LEVEL_DIFFERENCE < self.instance.tasks_difficulties[index_j]:
                        m.addConstr((x[i,j,k] == 0), name=f"no_assignment_if_level_diff_greater_than_1_{i}_{j}_{k}") 

    def _worker_of_the_first_operation_must_do_all_operations_of_the_job(self, m, x, y, job_done, task_done):

        # y_ik variable qui modélise si un worker est un worker de base pour le job i

        for i in range(self.instance.nb_jobs):
            # Un seul worker de base par job : garantit qu'un même worker suit toutes les opérations faites
            m.addConstr(gp.quicksum(y[i,k] for k in range(self.instance.nb_workers)) <= 1, name=f"at_most_one_base_worker_{i}")
            for j in range(len(self.instance.jobs_struct[i])):
                # Une opération ne peut être faite que si un worker de base est désigné pour ce job
                m.addConstr(task_done[i,j] <= gp.quicksum(y[i,k] for k in range(self.instance.nb_workers)), name=f"worker_base_task_done_{i}_{j}")

        # Le worker de base doit être présent sur chaque opération effectuée, mais pas sur les opérations non faites.
        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((x[i,j,k] >= y[i,k] + task_done[i,j] - 1), name=f"same_worker_operation_{i}_{j}_{k}")

    # HARD CONSTRAINTS
    def _at_least_one_worker_with_level_greater_than_difficulty_of_task(self, m, x, penalty_levels, z_auxilary, job_completed_without_level): # -> les taches en apprentissage doivent etre fait avec une personne de niveau ????
        # constraint : level of worker k must be >= difficulty of the operation assigned to worker k
        #              if 2 workers are asssigned to the same operation, at least one of the two workers must have a level higher than the difficulty of the opeation
        M = 2
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j] 
                    
                    # if worker k assigned to the task (i,j) does not have the level required for the task
                    if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:

                        # then at most 2 workers can be assigned to the same task and at least one of the other workers assigned to the same task must have the level required for the task
                        m.addConstr((x[i,j,k] + gp.quicksum(x[i,j,k_prime] for k_prime in range(self.instance.nb_workers) if k_prime != k) >= M*(x[i,j,k] - 1) + 2), name=f"at_most_two_workers_{i}_{j}_{k}") # at most 2 workers can be assigned to the same task if k do the task and he dont have the levels for it
                        
                        # La contrainte suivante permet de modéliser que si cette tache est fait par x_ij et n'a pas le niveau alors l'autre personne avec lui doit l'avoir
                        m.addConstr((gp.quicksum(x[i,j,k_prime] * self.instance.levels_workers[k_prime][index_m] for k_prime in range(self.instance.nb_workers) if k_prime != k)) >= self.instance.tasks_difficulties[index_j] * x[i,j,k], name=f"at_least_one_worker_with_capacity_{i}_{j}_{k}") # if worker k do the sub op and he dont have the levels for it, at least one of the other workers assigned to the same sub-op must have the level for it


        # put at zero the variables can be take value for worker without levels
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):

                # no solo no level
                m.addConstr((z_auxilary[i,j,3] == 0), name=f"z_auxilary_solo_level_zero_{i}_{j}") # z_auxilary[i,j,3] must be 0 if the constraint is respected because we do not want solo without level to do the task
                m.addConstr((penalty_levels[i,j] == 0), name=f"penalty_levels_zero_{i}_{j}")# penalty_levels[i,j] must be 0 if the constraint is respected because we do not want worker without level to do the task
            m.addConstr((job_completed_without_level[i]== 0), name=f"job_completed_without_level_zero_{i}") # job_completed_without_level[i] must be 0 if the constraint is respected because we do not want worker without level to do the task

    # SOFT CONSTRAINTS
    def _at_least_one_worker_with_level_greater_than_difficulty_of_task_SOFT(self, m, x, penalty_levels, job_completed_without_level, z_auxilary): # -> SOFT
        # This function is for accepting that a task ca be done by a worker without the required level but the difference of level can not be too high than 1

        # we calculate all the workers that do no not have the level required for each task and we blocked them to be together on the same task
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
                        
        # if worker do not have level, he can do the task but with a penalty_levels
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j] 
                    if self.instance.levels_workers[k][index_m] < self.instance.tasks_difficulties[index_j]:
                        # peut faire la tache seul mais pénalité
                        M = 4
                        # si fait en apprentissage alors deux personnes et pas de penalité, si fait seul et pas le niveau le une penalité
    
                        # Contrainte que si fait en solo ou en solo sans level
                        m.addConstr((self.instance.levels_workers[k][index_m] * x[i,j,k] >= self.instance.tasks_difficulties[index_j] * x[i,j,k] - penalty_levels[i,j] - M *(1 - (z_auxilary[i,j,0] + z_auxilary[i,j,3]))), name=f"penalty_if_worker_without_level_do_task_{i}_{j}_{k}")

        # jut one job can be done by worker without the level required but with penalty
        # just a line of penaly_levels can be > 0, it is the job completed by worker without the level required
        # at most one job can be done by workers without the level required but with penalty
        m.addConstr((gp.quicksum(job_completed_without_level[i] for i in range(self.instance.nb_jobs)) <= self.nb_job_with_no_skills), name=f"job_completed_without_level")

        # just the line (task) of job selected can be completed, others must be 0 because we want just one job can be done by worker without the level required but with penalty
        for i in range(self.instance.nb_jobs):
            nb_op_in_job = len(self.instance.jobs_struct[i])
            m.addConstr((gp.quicksum(penalty_levels[i,j] for j in range(len(self.instance.jobs_struct[i]))) <= nb_op_in_job * job_completed_without_level[i]), name=f"at_most_one_job_without_level_{i}")

        # the difference of level can not be more than 1 if the task is done by a worker without the required level
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((penalty_levels[i,j] <= LEVEL_DIFFERENCE), name=f"penalty_levels_positive_{i}_{j}") # the level difference can not be more than 1 if the task is done by a worker without the required level

    # NE PRENS PAS ENCORE EN COMPTE TACHE SOLO SANS COMPETENCE (A FAIRE)
    def _teaching_effect_constraints(self, m, x, l, delta_lin_trick_teach_effect):
        M_1 = 1
        M_4 = 4
        M = 1000
        N_CONSTR = 3
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
 
                x1 = self.instance.levels_workers[k][metier] + all_task_m_learning*COEF_LEARNING
                x2 = self.instance.levels_workers[k][metier] + 1

                m.addConstr((l[k,metier] <= x1), name=f"learning_effect_w{k}_metier{metier}") # learning effect for metier metier
                m.addConstr((l[k,metier] <= x2), name=f"max_learning_effect_w{k}_metier{metier}") # max learning effect for metier 
                m.addConstr((l[k,metier] <= LEVEL_MAX), name=f"max_level_metier{metier}") # level of worker k for metier metier can not be more than 4 because the max difficulty of sub-op is 4
                
                # pas besoin de la contrainte suivante si on fait lineariosation du min
                # m.addConstr((l[k,metier] >= self.instance.levels_workers[k][metier]), name=f"min_level_metier{metier}") # level of worker k for metier metier can not be less than the initial level of worker k for metier 
                

                ##### linearisation du min pour que l[k,m] se fixe sur l'augmentation réelle de niveau et non pas un intervalle de 0 à l'augmentation maximale possible
                m.addConstr((l[k,metier] >= x1 - M * (1 - delta_lin_trick_teach_effect[k,metier,0])), name=f"linearization_learning_effect1_w{k}_metier{metier}")
                m.addConstr((l[k,metier] >= x2 - M_1 * (1 - delta_lin_trick_teach_effect[k,metier,1])), name=f"linearization_learning_effect2_w{k}_metier{metier}")
                m.addConstr((l[k,metier] >= LEVEL_MAX - M_4 * (1 - delta_lin_trick_teach_effect[k,metier,2])), name=f"linearization_learning_effect3_w{k}_metier{metier}")


                m.addConstr((gp.quicksum(delta_lin_trick_teach_effect[k,metier,i] for i in range(N_CONSTR)) == 1), name=f"linearization_learning_effect4_w{k}_metier{metier}")

    def _cognitive_load_constraints(self, m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, tab_count_tasks_has_tutor, tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, tab_count_tasks_with_collab, cognitive_load_total, l):
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
                
        print("EPS=", EPS)
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
                    tab_count_tasks_has_tutor[k][index_m] += is_tutor[i,j,k] * self.instance.tasks_difficulties[index_j] * COEF_W_EFF + is_tutor[i,j,k] * COEF_TUTOR * (LEVEL_MAX + 1 - self.instance.levels_workers[k][index_m]) # menatal workload géneré par la tache 

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
                    tab_count_tasks_has_apprenti[k][index_m] += is_apprenti[i,j,k] * self.instance.tasks_difficulties[index_j] * COEF_W_EFF + is_apprenti[i,j,k] * COEF_APPRENTI * (LEVEL_MAX + 1 - self.instance.levels_workers[k][index_m]) # menatal workload géneré par la tache tab_count_tasks_has_apprenti[k][index_m] += is_apprenti[i,j,k] # nombre de taches den tant que apprenti de metier m pour le worker k


        # contrainte pour calculer la charge cognitive des tuteurs lorsqu'il apprennent une tache à un apprenti
        m.addConstrs((cognitive_load_tutors[k,metier] == tab_count_tasks_has_tutor[k][metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="count_tutor_tasks")

        # contrainte pour calculer la charge cognitive des apprentis lorsqu'ils apprennent une tache avec un tuteur
        m.addConstrs((cognitive_load_apprentis[k,metier] == tab_count_tasks_has_apprenti[k][metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)), name="count_apprenti_tasks")

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
                    
                    tab_count_tasks_with_collab[k][index_m] += is_collab[i,j,k] * self.instance.tasks_difficulties[index_j] * COEF_W_EFF + is_collab[i,j,k] * COEF_COLLAB * (LEVEL_MAX + 1 - self.instance.levels_workers[k][index_m] )



        # EN pause 
        # # if worker do no task, then his mental workload increase
        # M_NB_TASKS = self.instance.nb_jobs * self.instance.max_nb_operations # borne sup
        # for k in range(self.instance.nb_workers):
        #     do_task = gp.quicksum(cognitive_load_total[k,metier] for metier in range(self.instance.nb_professions))
        #     x = - do_task * M_NB_TASKS + 1
        #     m.addConstr((cogntitive_load_doing_nothing[k] == 2*x), name=f"cognitive_load_doing_nothing_{k}") 
        #     # Si il a fait aucune tache alors == 2 
        #     # Si il a fait au moins une tache alors == 0




        for k in range(self.instance.nb_workers):
            for metier in range(self.instance.nb_professions):
                m.addConstr((cognitive_load_collaboration[k,metier] == tab_count_tasks_with_collab[k][metier]), name=f"count_collaboration_tasks_{k}_{metier}")

                ###### SUM OF COGNITIVE LOADS ######
                # J'ai ajouté nouveau tableau de variable, mais pourrait etre fait dans la fonction objectif directement en sommant les trois charges cognitives !!
                # A voir ce qui est plus simple pour la résolution du modèle
                m.addConstr((cognitive_load_total[k,metier] == cognitive_load_tutors[k,metier] + cognitive_load_collaboration[k,metier] + cognitive_load_apprentis[k,metier]), name=f"cognitive_load_total_{k}_{metier}")

   
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
                    m.addConstr((z_auxilary[i,j,0] == 0), name=f"all_collaboration_tasks_{i}_{j}_{k}")
                    m.addConstr((z_auxilary[i,j,3] == 0), name=f"all_collaboration_tasks_{i}_{j}_{k}")

    def _all_solo_tasks(self, m, z_auxilary):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((z_auxilary[i,j,1] == 0), name=f"all_solo_tasks_{i}_{j}_{k}")
                    m.addConstr((z_auxilary[i,j,2] == 0), name=f"all_solo_tasks_{i}_{j}_{k}")

    def _not_constrained_makespan():
        pass


    def _constrained_makespan(self, m, C_max, limit_makespan, penalty_makespan, penalty_makespan_job, C, job_done):
        # fenetre de temps disponible pour faire les jobs
        self.BORNE_SUP_MAKESPAN = limit_makespan
        print("limite MAKESPAN = ", limit_makespan)
        m.addConstr((C_max <= limit_makespan), name=f"constrained_makespan")


    def _no_constrained_makespan(self, m, job_done):
        pass


    # CONSTRAINTS FOR FIXING THE VALUE OF OBJECTIVES FOR EPSILON CONSTRAINT METHOD
    def _fix_value_skills_superior(self, m, l, skills_value):
        m.addConstr((gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) - gp.quicksum(self.instance.levels_workers[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) >= skills_value), name=f"fix_value_skills_superior_eps_constr")

    def _fix_value_cognitive_load(self, m, cognitive_load_total, cognitive_load_value):
        m.addConstr((gp.quicksum(cognitive_load_total[k,m] for k in range(self.instance.nb_workers) for m in range(self.instance.nb_professions)) <= cognitive_load_value), name=f"fix_value_cognitive_load_eps_constr")


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
        
    def _build_variables(self, m) -> tuple:

        x = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="x") # x[i, j, k] = 1 if operation j of job i is assigned to worker k
        d = m.addVars(self.indexes["assignment"], vtype=GRB.CONTINUOUS, lb=0, name="d") # d[i, j, k] = starting time of operation j of job i if assigned to worker k
        C = m.addVars(self.instance.nb_jobs, vtype=GRB.CONTINUOUS, lb=0, name="C") # C[i] = completion time of job i if minimize sum of C[i], completion time of all jobs if minimize C_max
        C_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max") # C_max = makespan
        delta = m.addVars(self.indexes["sequencing"], vtype=GRB.BINARY, name="delta") # delta[i, j, h, g, k] = 1
        
        # WORKER DE BASE POUR UN JOB
        y = m.addVars(self.instance.nb_jobs, self.instance.nb_workers, vtype=GRB.BINARY, name="y") # y[i,k]=1 si k est worker de base pour jobe i, 0 sinon 
         
        # learning AND forgetting effects variables
        l = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="l") # l[k,m] = level of worker k before performing metier m after run of the PL
        # forgetting = m.addVars(self.instance.nb_workers, self.instance.nb_professions, lb=0, vtype=GRB.CONTINUOUS, name="forgetting") # forgetting[k,m] = niveau de forgetting pour le worker k et le metier m, utilisé pour modéliser les effets d'oublie
        
        f = m.addVars(self.indexes["assignment"], vtype=GRB.CONTINUOUS, lb=0, name="f") # f[i,j,k] = completion time of operation j of job i if assigned to worker k -- Ajout de cette variable pour prendre en compte le fait que la duré d'une tache peut etre different selon si fait en solo, en apprentissage ou en collab
        z_auxilary = m.addVars(self.indexes["mode"], vtype=GRB.BINARY, name="z_auxilary") # z[i,j,0] = 1 if O_ij is done in solo, z[i,j,1] = 1 if O_ij is done in apprentissage

        # Linearisation min pour savoir si une tache est fait en apprentissage ou en collab
        Level_min = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS, lb=0, name="Level_min") #vaut le level min d'un worker sur O_ij
        Delta_min = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="Delta_min") # pour linearisation du min


        # Linéarisation du min pour forcer l[k,m] à prendre la valeur de l'augmentation de niveau réel et non pas un intervalle de 0 à l'augmentation maximale possible
        delta_lin_trick_teach_effect = m.addVars(self.instance.nb_workers, self.instance.nb_professions, 3, vtype=GRB.BINARY, name="delta_lin_trick_teach_effect")



        # Pour la partie Ergonomie (Faire augmenter le niveau de fatigue cognitif des workers qui enseignent des taches pour lesquelles ils ont le niveau requis)
        
        ## tutor part
        is_tutor = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_tutor") # is_tutor[i,j,k] = 1 si O_ij est fait par k avec un apprenti
        has_level = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="has_level") # has_level[i,j,k] = 1 si k à le niveau de compétence requis pour faire O_ij
        cognitive_load_tutors = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="cognitive_load_tutors") # cognitive_load_tutors[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier avec un apprenti
        
        ## apprenti part
        is_apprenti = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_apprenti") # is_apprenti[i,j,k] = 1 si O_ij est fait par k en apprentissage
        cognitive_load_apprentis = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="cognitive_load_apprentis") # cognitive_load_apprentis[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier en apprentissage

        ## collaboration part
        is_collab = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="is_collab") # is_collab[i,j,k] = 1 si O_ij est fait par k en collaboration
        cognitive_load_collaboration = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="cognitive_load_collaboration") # cognitive_load_collaboration[k,m] = charge cognitive de worker k pour le métier m si il fait une tache de ce metier en collaboration

        ## load tutor + load collab + load apprenti
        cognitive_load_total = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="cognitive_load_total") # somme des charge cognitive de tutorat et de collaboration pour chaque worker et chaque métier, utilisé pour l'objectif de minimisation de la charge cognitive
        

        ## Pénalité pour les taches qui dépassent la durée de la période courante - Si toutes les taches ne peuvent être réalisées dans la période considérée
        # si une tache dépasse la durée de la période courante, elle est considéré comme une pénalité dans le calcul du makespan et pourra être traité dans une autre période
        penalty_makespan = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="penalty_makespan") # pénalité pour les taches qui dépassent la durée de la période courante, utilisé pour le multi-période
        penalty_makespan_job = m.addVars(self.instance.nb_jobs, vtype=GRB.CONTINUOUS, lb=0, name="penalty_makespan_job") # pénalité qu'engendre un job i a finir avant la deadline voulu si constrained makespan est actif 
        penalty_deadline = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS, lb=0, name="penalty_deadline") # pénalité pour les taches qui sont éxécuté pendant la date limite BORNE_SUP_MAKESPAN

        # SOFT CONSTRAINTS OF LEVELS
        penalty_levels = m.addVars(self.indexes["operation"], vtype=GRB.CONTINUOUS , lb=0, name="penalty_levels") # pénalité pour chaque tache fait en solo par un worker non compétent

        # JOB COMPLETED WITHOUT LEVEL if activate
        job_completed_without_level = m.addVars(self.instance.nb_jobs, vtype=GRB.BINARY, name="job_completed_without_level") # variable pour savoir si un job est complété alors qu'il avait des opérations faites par des workers non compétent de niveau de difference au plus 1 sur les opérations de ce job

        # KNOW WICH JOBS ARE DONE
        job_done = m.addVars(self.instance.nb_jobs, vtype=GRB.BINARY, name="job_done") # for maximizing the number of jobs done wirh their resale value
        task_done = m.addVars(self.indexes["operation"], vtype=GRB.BINARY, name="task_done") # for know if all the tasks of a job are completed
        # job_done_before_limit = m.addVars(self.instance.nb_jobs, vtype=GRB.BINARY, name="job_done_before_limit") # for maximizing the number of jobs done before the deadline

        # Variable permettant de savoir si une tache à débuter avant la date limite BORNE_SUP_MAKESPAN
        # Pour faire en sorte que cette tache doit se terminer avant la date limite BORNE_SUP_MAKESPAN ou alors si elle dépasse cette date ajouté en pénalité
        in_time = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="in_time") # in_time[i,j,k] = 1 si O_ij commence avant la date limite BORNE_SUP_MAKESPAN

        return x, d, C, C_max, delta, l, f, z_auxilary, Level_min, Delta_min, is_tutor, has_level, cognitive_load_tutors, is_apprenti, cognitive_load_apprentis, is_collab, cognitive_load_collaboration, cognitive_load_total, penalty_makespan, penalty_deadline, in_time, penalty_levels, job_completed_without_level, job_done, task_done, y, delta_lin_trick_teach_effect, penalty_makespan_job #, forgetting

    def _build_helper_variables(self) -> None:
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
    
    def _build_model(
            self,
            objective,
            weight,
            priority,
            time_limit=None,
            constraints_config= None,
            verbose=False,
            job_with_no_skills=False,
            agregation_skills="sum",
            ponderation_task_done="one",
    ) -> gp.Model:
        """
        Construit le modèle de programmation linéaire
        
        Args:
            objective (int): l'objectif à optimiser, peut être "lexicographic" ou "three"
            weight (dico): les poids à accorder à chaque objectifs (benefit, skill, cognitive_load) si objective = "three", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (benefit, skill, cognitive_load) si objective = "three", n'est pas utilisé sinon
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            time_limit (int): la limite de temps pour l'optimisation

        Returns:
            m (gp.Model): le modèle de programmation linéaire construit
        """
        self.m = gp.Model(f"dual_resource_scheduling_objective_{objective}")
        m = self.m

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
            penalty_levels,
            job_completed_without_level,
            job_done,
            task_done,
            y,
            delta_lin_trick_teach_effect,
            penalty_makespan_job
            # forgetting
        ) = self._build_variables(m)

        self._build_helper_variables()

        # constraints
        # voir si meilleure de mettre ces conditions dans une fonction _build_constraints
        self._add_physical_constraints(m, x, d, C, C_max, z_auxilary, Level_min, Delta_min, f, delta, job_done, task_done)
        self._teaching_effect_constraints(m, x, l, delta_lin_trick_teach_effect)
        self._cognitive_load_constraints(m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, self.tab_count_tasks_has_tutor, self.tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, self.tab_count_tasks_with_collab, cognitive_load_total, l)
                

        if constraints_config is not None:

            # if a worker do not have the level for complete a task, a tutor can teaching him, or he can complete the task alone but with more time
            if constraints_config.get("job_with_no_skills", False) :
                print("job with no skills activated")
                self.nb_job_with_no_skills = constraints_config["job_with_no_skills"]
                print("nb_job_with_no_skills = ", self.nb_job_with_no_skills)
                self._at_least_one_worker_with_level_greater_than_difficulty_of_task_SOFT(m, x, penalty_levels, job_completed_without_level, z_auxilary)
                self.PENALTY_LEVEL = True
            else :
                print("job with no skills desactived")
                self.nb_job_with_no_skills = 0
                print("nb_job_with_no_skills = ", self.nb_job_with_no_skills)
                self._at_least_one_worker_with_level_greater_than_difficulty_of_task(m, x, penalty_levels, z_auxilary, job_completed_without_level)
                self.PENALTY_LEVEL = False
            

            if constraints_config.get("no_base_worker", False) :
                pass
            else :
                self._worker_of_the_first_operation_must_do_all_operations_of_the_job(m, x, y, job_done, task_done)


            # for epsilon constraint
            if constraints_config.get("fix_value_skills_superior", False) :
                skills_value = constraints_config["fix_value_skills_superior"]
                self._fix_value_skills_superior(m, l, skills_value)
    
            # for epsilon constraint
            if constraints_config.get("fix_value_cognitive_load", False) :
                cognitive_load_value = constraints_config["fix_value_cognitive_load"]
                self._fix_value_cognitive_load(m, cognitive_load_total, cognitive_load_value)


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
                limit_makespan = constraints_config["constrained_makespan"]
                self._constrained_makespan(m, C_max, limit_makespan, penalty_makespan, penalty_makespan_job, C, job_done)
                print("constrained makespan activated with limit =", limit_makespan)
                print("constrained makespan activated with self.BORNE_SUP_MAKESPAN =", self.BORNE_SUP_MAKESPAN)
            

            # If need to init affectation variables for some values
            init_vars = constraints_config.get("init_affectation_variables", False)
            if type(init_vars) is list :
                print("init affectation variables activated")
                self.init_affectation_variables(x, init_vars[0], d, init_vars[1], task_done, init_vars[2], z_auxilary, init_vars[3])
            else:
                print("init affectation variables desactivated")


        else : 
            print("job with no skills desactived")
            self._at_least_one_worker_with_level_greater_than_difficulty_of_task(m, x, penalty_levels, z_auxilary, job_completed_without_level)
            self.PENALTY_LEVEL = False
        

        ########################################################################
        ########################### OBJECTIVE FUNCTION #########################
        ########################################################################
        if time_limit is not None:
            m.Params.TimeLimit = time_limit

        m.Params.SolFiles = "../results/intermediate_solutions.sol"



        ###==================================== ==================================== ====================================
        ###====================================   DIFFERENT MODE D'OPTIM LE SKILLS   ====================================
        ###==================================== ==================================== ====================================
        
        if agregation_skills == "sum":
            skill_obj = gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) - gp.quicksum(self.instance.levels_workers[k][metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        
        elif agregation_skills == "max_min":
            for k in range(self.instance.nb_workers):
                for metier in range(self.instance.nb_professions):
                    m.addConstr((l[k,metier] - self.instance.levels_workers[k][metier] >= skill_obj), name=f"definition_z_max_min_{k}_{metier}")
        # elif agregation_skills == "max_min_line":
        # elif agregation_skills == "max_min_column":


        cognitive_load_total_obj = gp.quicksum(cognitive_load_total[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))

        # Job complété
        job_done_benefit = gp.quicksum(self.instance.resale_price_jobs[i] * job_done[i] for i in range(self.instance.nb_jobs))        

        # task complété pondéré par difficulté de la tache 
        if ponderation_task_done == "difficulty":
            task_done_benefit = gp.quicksum(self.instance.get_difficulty_and_metier_of_task(i, j)[0] * task_done[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
        elif ponderation_task_done == "one":
            task_done_benefit = gp.quicksum(task_done[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
        else:
            raise ValueError("ponderation_task_done doit être 'difficulty' ou 'one'")
    
        obj_benefit = job_done_benefit + task_done_benefit

        # ===============================================
        # =========== LEXICOGRAPHIC OBJECTIVE ===========
        # ===============================================

        if objective == "lexicographic": # valeur élevé implique priorité élevé
            m.setObjectiveN(obj_benefit , index=0, priority=priority[0], name="maximimize_benefit_obj")
            m.setObjectiveN(skill_obj, index=1, priority=priority[1], name="maximize_skill_levels_obj")
            m.setObjectiveN(-cognitive_load_total_obj, index=2, priority=priority[2], name="maximize_minus_cognitive_load_total_obj")
            m.modelSense = GRB.MAXIMIZE

        elif objective == "three":
            res = obj_benefit*weight["profit"] + skill_obj*weight["skills"] - cognitive_load_total_obj*weight["cognitive_load"] 
            m.setObjective(res, GRB.MAXIMIZE)

        else:
            raise ValueError("objective doit être 'makespan', 'skill', 'three', 'lexicographic' ou 'cognitive_load_tutors'")
        
        m.write(f"../results/model_{objective}.lp")
        return m
    
    def solve(self, objective="benefit" , weight={"profit": 0, "skills": 0, "cognitive_load": 0}, priority=[0,1,2], time_limit=None, constraints_config=None, verbose=False, job_with_no_skills=False, agregation_skills="sum") -> Solution:
        """
        Résout le modèle et affiche les résultats
        
        Args:
            objective (str): l'objectif à optimiser, peut être "lexicographic" ou "three"
            weight (dict): les poids à accorder à chaque objectifs (benefit, skill, cognitive_load) si objective = "three", n'est pas utilisé sinon
            priority (list): la priorité à accorder à chaque objectif (benefit, skill, cognitive_load_total) si objective = "lexicographic", n'est pas utilisé sinon
            time_limit (int): la limite de temps pour l'optimisation
            constraints_config (dict): la configuration des contraintes à ajouter au modèle, par exemple {"no_teaching_tasks": True, "constrained_makespan": True, ...}
            verbose (bool): si True, affiche les informations sur les solutions trouvées par Gurobi
            
        Returns:
            (Solution): une instance de la classe Solution contenant les résultats de la résolution du modèle
        """

        
        assert objective in ["lexicographic", "three"], "objective doit être 'lexicographic' ou 'three' "
        assert len(weight) == 3, "weight doit être un dictionnaire de trois éléments"
        assert len(priority) == 3, "priority doit être une liste de trois éléments"
        
        print("\n ========== SOLVING MODEL ========== ")
        m = self._build_model(
            objective, 
            weight, 
            priority, 
            time_limit, 
            constraints_config, 
            verbose, 
            job_with_no_skills, 
            agregation_skills=agregation_skills
        )

        if verbose == False:
            m.setParam('OutputFlag', 0) # to disable gurobi output

        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found with objective value:", m.objVal)
            m.write("../results/solution.sol")
            self.write_objectives_values(m, m.NumObj, "../results/objectives_values.txt")
        
        elif m.status == GRB.INF_OR_UNBD:
            print("Model is infeasible. Status code:", m.status)
            m.computeIIS()
            m.write("../results/model_iis.ilp")
            
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
        else :
            res.append(('Obj0', m.objVal))

        
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

        res.append(("BORNE_SUP_MAKESPAN", self.BORNE_SUP_MAKESPAN)) if self.BORNE_SUP_MAKESPAN is not None else res.append(("BORNE_SUP_MAKESPAN", -1))
        
        if verbose :
            print("objective value:", m.objVal)
            print(res)

        s = Solution(self.instance)
        s.from_milp_var_list(res)
        return s    

    def init_affectation_variables(self, x, x_init, d, d_init, task_done, task_done_init, z, z_init):
        """ Depart des valeurs de la variable x pour la resolution aux valeurs de x, d, task_done et z initiales passées en paramètre """
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    x[i,j,k].start = x_init[i,j,k]
                    d[i,j,k].start = d_init[i,j,k]
                task_done[i,j].start = task_done_init[i,j]
                for m in range(4):
                    z[i,j,m].start = z_init[i,j,m]



        print("$*$*$*$*$*$ initial affectation variables set to x_initial $*$*$*$*$*$")

"""
    def epsilon_contraintes(self, constraints_config=None, verbose=False):

        # résolution lexicogrpahique 
        solution_list = [] # liste des solutions obtenues pour chaque ordre d'optimisation lexicographique
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[2, 1, 0], verbose=verbose))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[2, 0, 1], verbose=verbose))

        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[1, 2, 0], verbose=verbose))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[0, 2, 1], verbose=verbose))

        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[0, 1, 2], verbose=verbose))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[1, 0, 2], verbose=verbose))



        # structure en numpy
        x = np.zeros((len(solution_list), 3)) 
        for i in range (len(x)):
            # for j in range(len(x[i])):
            #     x[i][j] = solution_list[i].objective_values[j]
            x[i][0] = solution_list[i].resale_price_job_done_before_limit() if benefit_in_time == True else solution_list[i].resale_price_job_done()
            x[i][1] = solution_list[i].sum_skill_levels_rate()
            x[i][2] = - solution_list[i].sum_cognitive_load_total()



        print("solutions for different lexicographic priorities:")
        print(x)
        # find best value for each objective
        best_benefit = max(x[:,0])
        best_skill = max(x[:,1])
        bad_cognitive_load = -min(x[:,2]) # je le tranforme en positif car dans mes contraintes je considère que la charge cognitive [0, +inf[
        
        # fix arange for the 2 objectives
        # skill_arange = np.arange(0, best_skill, 0.1)
        # cognitive_load_arange = np.arange(bad_cognitive_load, 0, -2)

        skill_arange = np.linspace(0, best_skill, 3)
        cognitive_load_arange = np.linspace(bad_cognitive_load, 0, 3)

        print("skill_arange:", skill_arange)
        print("cognitive_load_arange:", cognitive_load_arange)

        print("===========================================")
        print("======== BEGIN EPSILON CONTRAINTS ======== ")
        print("===========================================")
        
        not_solved = 0

        for skill_value in skill_arange:
            for cognitive_load_value in cognitive_load_arange:
                print("skill_value:", skill_value, "cognitive_load_value:", cognitive_load_value)

                constraints_config_epsilon = constraints_config.copy() if constraints_config is not None else {}
                # print("constraints_config_epsilon avant:", constraints_config_epsilon)
                constraints_config_epsilon["fix_value_skills_superior"] = skill_value
                constraints_config_epsilon["fix_value_cognitive_load"] = cognitive_load_value
                # print("constraints_config_epsilon après:", constraints_config_epsilon)
                
                # s = self.solve(objective="lexicographic", priority=[2, 1, 0], constraints_config=constraints_config_epsilon, verbose=verbose)
                
                # utilisé cette objectif plutot
                s = self.solve(objective="benefit", priority=[2, 1, 0], constraints_config=constraints_config_epsilon, verbose=verbose)
                print("Maj : benefit objective")

                if s is not None:
                    print("epsilon_skill:", skill_value, "epsilon_cognitive_load:", cognitive_load_value, "benefit:", s.objective_values[0])
                    

                    # x1 = s.objective_values[0]
                    # x2 = np.sum(s.l) - np.sum(self.instance.levels_workers)
                    # x3 = - np.sum(s.cognitive_load_total)
                    
                    x1 = s.resale_price_job_done_before_limit() if benefit_in_time == True else s.resale_price_job_done()
                    x2 = s.sum_skill_levels_rate()
                    x3 = - s.sum_cognitive_load_total()

                    print("x1:", x1, "x2:", x2, "x3:", x3)
                    y = np.array([x1, x2, x3])
                    x = np.append(x, y).reshape(-1, 3)
                    solution_list.append(s)
                else :
                    print("No solution found for epsilon_skill:", skill_value, "epsilon_cognitive_load:", cognitive_load_value)
                    not_solved += 1

        print("number of not solved epsilon constraints:", not_solved)

        print("===========================================")
        print("======== END EPSILON CONTRAINTS ========")
        print("===========================================")



        # if plot:
        #     plt.figure("Exemple 3D")
        #     axes = plt.axes(projection="3d")
        #     axes.set_xlabel("profit")
        #     axes.set_ylabel("skills")
        #     axes.set_zlabel("- cognitive_load")

        #     for i in range(len(x)):
        #         axes.scatter(x[i,0], x[i,1], x[i,2], marker="o")
        #     plt.show()
           
        return x, solution_list

"""

if __name__ == "__main__":
    

    # #======== Instance Creation ========
    # res = read_file("../data/data_resale_price_hand_solving_2.test")
    # instance = Instance()
    # instance.from_dictionary(res)
    # print(instance)
    # instance.qualified_workers_for_task(verbose=True)

############################
############################

    time_type_of_jobs = {
        "small": (3, 9),
        "medium": (9, 18),
        "long": (18, 42)
    }

    # seed = np.random.randint(1, 1000) # pour générer un nombre aléatoire entre 1 et 1000 (inclus)
    seed = 135
    print(f"seed : {seed}")
    config_instance = {
        "seed": seed,

        "nb_jobs": 3,
        "nb_workers": 3,
        "nb_professions": 1,
        "proportion_tasks_per_profession": [1], # [profession 1, profession 2]
        "max_nb_operations": 5,
        "nb_tasks" : "lower", # "lower", "medium", "high"

        # "proportion_jobs": [0.3, 0.4, 0.3], # [small_jobs, medium_jobs, long_jobs]

        "proportion_tasks_without_collaborative_work": 0.2,
        "proportion_tasks_level_required": 0.1,

        "proportion_tasks_times" : [0.3, 0.4, 0.3], # [small_tasks (30 unités), medium_tasks (), long_tasks]
        "time_type_of_jobs" : time_type_of_jobs,
        "proportion_tasks_levels":[ 0.1, 0.3, 0.5, 0.1], # [level 1, level 2, level 3, level 4]

        "proportion_workers_levels":[ 0.2, 0.5, 0.2, 0.1], # [level 1, level 2, level 3, level 4]

        "at_least_a_worker_have_competence_for_each_profession": False
    }
    instance = Instance()
    instance.from_config(config_instance)
    instance.tasks_times = np.array(instance.tasks_times, dtype=int)
    print(instance)
    # visualization_before_scheduling(instance, not_qualified=True)

    TIME_LIMIT_MAKESPAN = instance.sum_of_job_alone() / 2
    print("temps total laissé pour le scheduling :", TIME_LIMIT_MAKESPAN)
    print("temps total nécessaire pour faire tous les jobs en travaillant seul :", instance.sum_of_job_alone())

    
    #======== Model Creation ========
    model = Model(instance)
    
    # TIME_LIMIT_MAKESPAN = instance.sum_of_job_alone()#/ 2
    constraints_config = {}
    constraints_config = {"job_with_no_skills": True,
                        "constrained_makespan": TIME_LIMIT_MAKESPAN,
                        "no_base_worker" : False,
                        "no_solo_tasks": False,
                 }

    #======== Dico de pondération des objectifs (Aggrégation) ========
    weights = {
    "profit" : 1,
    "skills" : 0,
    "cognitive_load" : 0
    }



    s = model.solve(objective="three", weight=weights, priority=[2, 1, 0], constraints_config=constraints_config, verbose=True )
    # resume_levels_workers(s, instance)
    print(s)


    #======== Solution Visualization ========
    df = scheduling_to_df(s, instance)
    gantt_chart(df, color=0, separate_little=True)
    print(df)

    res = check_df(df)
    print("check_df:", res)

    res = s.all_jobs_completed()
    print("all jobs completed:", res)

    res = s.which_jobs_are_completed()
    print("completed jobs:", res)

     
