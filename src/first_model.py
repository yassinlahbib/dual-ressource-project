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
    
        # constraint :
        # At most 2 workers per tasks
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])): # for all operations except the first one of each job
                m.addConstr((gp.quicksum(x[i, j, k] for k in range(self.instance.nb_workers)) <= 2), name=f"max_assignment_operation_{i}_{j}")
        

    #    # NOW : tasks can be not assigned if they are too difficult or if he do not respect limit Cmax time
    #     # constraint : 
    #     # All operations must be assigned to at least one worker
    #     for i in range(self.instance.nb_jobs):
    #         for j in range(len(self.instance.jobs_struct[i])):
    #             m.addConstr((gp.quicksum(x[i, j, k] for k in range(self.instance.nb_workers)) >= 1), name=f"min_sub_operation_assignment_{i}_{j}")


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
                m.addConstr((z_auxilary[i,j,0] + z_auxilary[i,j,1] + z_auxilary[i,j,2] + z_auxilary[i,j,3] <= 1), name=f"z_assignment_1_{i}_{j}") # before it was == 1 but i consider all task are assinged, but not now, we can have tasks not assigned
                # fixé z_ijs0 à 1 si O_ijs est fait en solo_level ou solo sans level.   z_ijs1 à 1 si O_ijs est fait en apprentissage, z_ijs2 à 1 si O_ijs est fait en collab
                m.addConstr(((gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers)) == 1 * z_auxilary[i,j,0] + 1 * z_auxilary[i,j,3] + 2 * z_auxilary[i,j,1] + 2 * z_auxilary[i,j,2])), name=f"z_assignment_2_{i}_{j}")
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
                # comme une tache peut ne pas être affécter alors passer de "== 1" à "<= 1"
                m.addConstr((gp.quicksum(Delta_min[i,j,k] for k in range(self.instance.nb_workers)) <= 1), name=f"linearization_min_binary_{i}_{j}") # Delta_min_ijsk doit prendre 1 pour le k tq il a le level minimal pour cette tache
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
                    # m.addConstr((f[i,j,k] <= time), name=f"definition_f_{i}_{j}_{k}") # s'il fait la tache f <= d + times, s'il la fait pas f est libre AVOIR SI DERANGEANT
                    # contraintes a verif !!!!!!
                    # m.addConstr((f[i,j,k] >= time - M * (1 - x[i,j,k])), name=f"definition_f2_{i}_{j}_{k}") 
                    m.addConstr((f[i,j,k] == time), name=f"definition_f3_{i}_{j}_{k}") # si x[i,j,k] = 1 alors f[i,j,k] = d[i,j,k] + time, sinon f[i,j,k] est libre mais doit respecter les autres contraintes du modeles pour ne pas être trop grand

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

        # Si toutes les taches sont affecté alors correcte, mais comme on peut avoir des taches non affecté
        # alors il y a un problème car si tache 1 avant tache 2 et tache 1 pas affecté alors tache 2 peut être affecté et d2 > f1 = 0 alors que tache 1 doit être avant tache 2
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
                                    m.addConstr((f_ijk <= M * (1 - x[i,j_prime,k_prime]) + d[i,j_prime,k_prime]  ), name=f"precedence_operations_inactive_{i}_{j}_{j_prime}_{k}_{k_prime}")

                            # ajout du fait que la tache i+1 doit etre effectué que si la tache i est effectué pour éviter les problèmes de tache non affecté et de precedence
                            # car maintenant des taches peuvent ne pas etre affecté 
                            m.addConstr((task_done[i,j] >= task_done[i,j_prime]), name=f"precedence_operations_task_done_{i}_{j}_{j_prime}")

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

        # constraint :
        # if task (i,j) is assigned to at least one worker then task_done[i,j] = 1, else task_done[i,j] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((task_done[i,j] <= gp.quicksum(x[i,j,k] for k in range(self.instance.nb_workers))), name=f"task_done_definition_{i}_{j}")

        # constraint :
        # if all tasks of a job i are done then job_done[i] = 1, else job_done[i] = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((job_done[i] <= task_done[i,j]), name=f"job_done_definition_{i}_{j}") # job_done[i] = 1 if all tasks of job i are done, else job_done[i] = 0
        


        # si job done = 0 alors alors ne pas affecté les tache de ce job :
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((x[i,j,k] <= job_done[i]), name=f"no_assignment_if_job_not_done_{i}_{j}_{k}") # if job is not done then no task of this job can be assigned



        # Un opérateur ne peut faire une tache en tant qu'apprenti si son niveau de différence avec la tache est >1
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    index_j = self.instance.jobs_struct[i][j]
                    index_m = self.instance.task_to_m[index_j]
                    if self.instance.levels_workers[k][index_m] + 1 < self.instance.tasks_difficulties[index_j]:
                        m.addConstr((x[i,j,k] == 0), name=f"no_assignment_if_level_diff_greater_than_1_{i}_{j}_{k}") 
                        # en mettant la conteainte suivante, j'interdis la tache ij d'être fait en apprentissage, si un worker n' a pas le niveau 
                        # suffisant pour faire la tache, alors que d'autre worker peuvent la faire en tant qu'apprentis sa se trouve (mieux vaux tard que jamais pour s'en rendre compte)
                        
                        # m.addConstr((z_auxilary[i,j,1] == 0), name=f"no_apprenticeship_if_level_diff_greater_than_1_{i}_{j}_{k}")

    def _worker_of_the_first_operation_must_do_all_operations_of_the_job(self, m, x, y, job_done):

        # HYPOTHESE :
        # La première opération de chaque job doit être réalisé par un seul worker.
        # Si ce n'est pas possible alors ajouter une tache fictive au début du job pour l'affecter au worker qui sera assigné à toutes les opération du job en question

        # 1ere idée:
        # La premiere opeariotn de chaque job doit etre réaliser par un seul worker pour pouvoir l'assigner a toutes les opération suivante de ce job
        # constraint :
        # For the first operation of each job at most 1 worker for we know it is the worker assigned to the all job
        # for i in range(self.instance.nb_jobs):
        #     m.addConstr((gp.quicksum(x[i, 0, k] for k in range(self.instance.nb_workers)) <= 1), name=f"max_first_sub_operation_assignment_{i}")


        # 2eme idée:
        # avoir une varibale base y_ik qui modélise si un worker est un worker de base pour le job i

        # Si le job est terminé alors un worker de base à du etre dessus
        # Cette contrainte force à ne choisir qu'un worker k par y_ik
        for i in range(self.instance.nb_jobs):
            m.addConstr((job_done[i] <= gp.quicksum(y[i,k] for k in range(self.instance.nb_workers))), name=f"worker_base_job_done_{i}") 

        # Ce worker choisit doit avoir fait tout les opérationd de ce job
        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    m.addConstr((x[i,j,k] >= y[i,k]), name=f"same_worker_operation_{i}_{j}_{k}")

        # Plus besoin de cette contrainte à present
        # # constraint : if worker start the first task of a job then this worker must do all the tasks for this job
        # for i in range(self.instance.nb_jobs):
        #     for j in range(len(self.instance.jobs_struct[i])):
        #         for k in range(self.instance.nb_workers):
        #             # print("ici->", i, j, k)
        #             m.addConstr((x[i,0,k] <= x[i,j,k]), name=f"same_worker_operation_{i}_{j}_{k}")

    def _at_least_one_worker_with_level_greater_than_difficulty_of_task(self, m, x, penalty_levels, z_auxilary, job_completed_without_level): # -> les taches en apprentissage doivent etre fait avec une personne de niveau ????
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


        # put at zero the variables can be take value for worker without levels
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
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
                        
                        # Avant
                        # m.addConstr((self.instance.levels_workers[k][index_m] * x[i,j,k] >= self.instance.tasks_difficulties[index_j] * x[i,j,k] - penalty_levels[i,j] - M *(1 - gp.quicksum(x[i,j,kprime] for kprime in range(self.instance.nb_workers)))), name=f"penalty_if_worker_without_level_do_task_{i}_{j}_{k}")
                        
                        #maintenant : Contraint que si fait en solo ou en solo sans level
                        m.addConstr((self.instance.levels_workers[k][index_m] * x[i,j,k] >= self.instance.tasks_difficulties[index_j] * x[i,j,k] - penalty_levels[i,j] - M *(1 - (z_auxilary[i,j,0] + z_auxilary[i,j,3]))), name=f"penalty_if_worker_without_level_do_task_{i}_{j}_{k}")

        # jut one job can be done by worker without the level required but with penalty
        # just a line of penaly_levels can be > 0, it is the job completed by worker without the level required

        # at most one job can be done by workers without the level required but with penalty
        m.addConstr((gp.quicksum(job_completed_without_level[i] for i in range(self.instance.nb_jobs)) <= 1), name=f"job_completed_without_level")

        # just the line (task) of job selected can be completed, others must be 0 because we want just one job can be done by worker without the level required but with penalty
        for i in range(self.instance.nb_jobs):
            nb_op_in_job = len(self.instance.jobs_struct[i])
            m.addConstr((gp.quicksum(penalty_levels[i,j] for j in range(len(self.instance.jobs_struct[i]))) <= nb_op_in_job * job_completed_without_level[i]), name=f"at_most_one_job_without_level_{i}")

        # the difference of level can not be more than 1 if the task is done by a worker without the required level
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((penalty_levels[i,j] <= 1), name=f"penalty_levels_positive_{i}_{j}") # the level difference can not be more than 1 if the task is done by a worker without the required level

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

    # OLD COGNITIF CONSTRAINT
    '''
    def _cognitive_load_constraints_OLD(self, m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, tab_count_tasks_has_tutor, tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, tab_count_tasks_with_collab, cognitive_load_total):

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
    '''
   
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


##### FONCTION A BINE REVOIR !!!!!!!!!!!!!! MAJ LE 5 MAI ET PAS TEMINER DONC A TEMRINER
    def _constrained_makespan(self, m, C_max, limit_makespan, penalty_makespan, penalty_makespan_job, job_done_before_limit, C, job_done):
        # if makespan is grater than a certain thresold, we consider penalty for the exceeded part
        self.BORNE_SUP_MAKESPAN = limit_makespan
        print("limite MAKESPAN = ", limit_makespan)
        m.addConstr((C_max <= limit_makespan + penalty_makespan), name=f"constrained_makespan")

        for i in range(self.instance.nb_jobs):
            m.addConstr((C[i] <= limit_makespan + penalty_makespan_job[i]))

        M = max(self.instance.tasks_times[i][0] for i in range(self.instance.nb_tasks)) * self.instance.max_nb_operations # borne sup du makespan
        for i in range(self.instance.nb_jobs):
            m.addConstr((penalty_makespan_job[i] <= (1 - job_done_before_limit[i]) * M))
            m.addConstr((job_done_before_limit[i] <= job_done[i]))


    # NE PRENS PAS ENCORE EN COMPTE TACHE SOLO SANS COMPETENCE (A FAIRE)
    def _deadline_constraints_operation(self, m, x, d, z_auxilary, in_time, penalty_deadline, C, type="job"):
        """
        type : "operation" ou "job" pour savoir si on applique la contrainte sur les opérations ou sur les jobs
        """
        ########### Une tache (considérer jobs ou opérations ou sous-opérations) qui commence avant la date limite BORNE_SUP_MAKESPAN doit se terminer avant celle ci
        ## -> forcer cette contrainte en dure pour le moment mais voir si on peut autoriser mais grande pénalité si on la dépasse pour ne pas rendre le modèle infaisable
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    
                    # in_time[i,j,k] = 1 alors d[i,j,k] <= BORNE_SUP_MAKESPAN, 0 sinon
                    # Si k ne fait pas la tache O_ijs -> in_time[i,j,k] = 1
                    m.addConstr((d[i,j,k] >= self.BORNE_SUP_MAKESPAN - M * in_time[i,j,k]), name=f"start_time_before_deadline_{i}_{j}_{k}")
                    m.addConstr((d[i,j,k] <= self.BORNE_SUP_MAKESPAN + M * (1 - in_time[i,j,k])), name=f"start_time_before_deadline2_{i}_{j}_{k}")

                    if type == "operation":
                        index_j = self.instance.jobs_struct[i][j] # A VOIR -> Besoin pour calculer la fin de la tache mais avec variable f_ijk directement quand je l'aurais terminer les contraintes sur cette varibale f_ijk
                        # si k fait la tache : f_ijsk vrai fin
                            # si in_time = 1 -> f_ijsk doit se terminer avant OK
                            # si in_time = 0 -> f_ijsk non contraint par cette coontrainte OK
                        # si k ne fait pas la tache : f_ijsk = - M <= BORNE_SUP_MAKESPAN donc OK

                        f_ijsk = d[i,j,k] + self.instance.tasks_times[index_j][0] * z_auxilary[i,j,0] + self.instance.tasks_times[index_j][1] * z_auxilary[i,j,1] + self.instance.tasks_times[index_j][2] * z_auxilary[i,j,2] - M * (1 - x[i,j,k])
                        m.addConstr((f_ijsk <= self.BORNE_SUP_MAKESPAN + penalty_deadline[i,j] + M * (1 - in_time[i,j,k])), name=f"end_time_before_deadline_{i}_{j}_{k}")

                    if type == "job":
                        m.addConstr((C[i] <= self.BORNE_SUP_MAKESPAN + gp.quicksum(penalty_deadline[i,j] * in_time[i,j,k] for j in range(len(self.instance.jobs_struct[i])))), name=f"job_completion_before_deadline_{i}_{j}_{k}")


        self.PENALTY_DEADLINE = True

    def _hard_constraints_level_must_be_higher_if_solo(self, m, z_auxilary, ):
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                m.addConstr((z_auxilary[i,j,3] == 0), name=f"no_solo_without_level_{i}_{j}")

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
        y = m.addVars(self.instance.nb_jobs, self.instance.nb_workers, vtype=GRB.INTEGER, name="y") # y[i,k]=1 si k est worker de base pour jobe i, 0 sinon 
         
        # learning AND forgetting effects variables
        l = m.addVars(self.instance.nb_workers, self.instance.nb_professions, vtype=GRB.CONTINUOUS, lb=0, name="l") # l[k,m] = level of worker k before performing metier m after run of the PL
        # forgetting = m.addVars(self.instance.nb_workers, self.instance.nb_professions, lb=0, vtype=GRB.CONTINUOUS, name="forgetting") # forgetting[k,m] = niveau de forgetting pour le worker k et le metier m, utilisé pour modéliser les effets d'oublie
        
        f = m.addVars(self.indexes["assignment"], vtype=GRB.CONTINUOUS, lb=0, name="f") # f[i,j,k] = completion time of operation j of job i if assigned to worker k -- Ajout de cette variable pour prendre en compte le fait que la duré d'une tache peut etre different selon si fait en solo, en apprentissage ou en collab
        z_auxilary = m.addVars(self.indexes["mode"], vtype=GRB.INTEGER, name="z_auxilary") # z[i,j,0] = 1 if O_ij is done in solo, z[i,j,1] = 1 if O_ij is done in apprentissage

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
        job_done_before_limit = m.addVars(self.instance.nb_jobs, vtype=GRB.BINARY, name="job_done_before_limit") # for maximizing the number of jobs done before the deadline

        # Variable permettant de savoir si une tache à débuter avant la date limite BORNE_SUP_MAKESPAN
        # Pour faire en sorte que cette tache doit se terminer avant la date limite BORNE_SUP_MAKESPAN ou alors si elle dépasse cette date ajouté en pénalité
        in_time = m.addVars(self.indexes["assignment"], vtype=GRB.BINARY, name="in_time") # in_time[i,j,k] = 1 si O_ij commence avant la date limite BORNE_SUP_MAKESPAN

        return x, d, C, C_max, delta, l, f, z_auxilary, Level_min, Delta_min, is_tutor, has_level, cognitive_load_tutors, is_apprenti, cognitive_load_apprentis, is_collab, cognitive_load_collaboration, cognitive_load_total, penalty_makespan, penalty_deadline, in_time, penalty_levels, job_completed_without_level, job_done, task_done, y, delta_lin_trick_teach_effect, penalty_makespan_job, job_done_before_limit #, forgetting

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
    
    def _build_model(self, objective, weight, priority, time_limit=None, constraints_config= None, verbose=False, job_with_no_skills=False, agregation_skills="sum", benefit_in_time=True) -> gp.Model:
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
            penalty_makespan_job,
            job_done_before_limit
            # forgetting
        ) = self._build_variables(m)

        self._build_helper_variables()

        # constraints

        # voir si meilleure de mettre ces conditions dans une fonction _build_constraints
        self._add_physical_constraints(m, x, d, C, C_max, z_auxilary, Level_min, Delta_min, f, delta, job_done, task_done)
        self._teaching_effect_constraints(m, x, l, delta_lin_trick_teach_effect)
        self._cognitive_load_constraints(m, x, z_auxilary, has_level, is_tutor, is_apprenti, is_collab, self.tab_count_tasks_has_tutor, self.tab_count_tasks_has_apprenti, cognitive_load_tutors, cognitive_load_apprentis, cognitive_load_collaboration, self.tab_count_tasks_with_collab, cognitive_load_total, l)
                
        



        if constraints_config is not None:

            if constraints_config.get("job_with_no_skills", False) :
                # if a worker do not have the level for complete a task, a tutor can teaching him, or he can complete the task alone but with more time and a penalty_levels on the objectif function is appliynig
                print("job with no skills activated")
                self._at_least_one_worker_with_level_greater_than_difficulty_of_task_SOFT(m, x, penalty_levels, job_completed_without_level, z_auxilary)
                self.PENALTY_LEVEL = True
            else :
                print("job with no skills desactived")
                # if a worker do not have level for complete a task, then a worker with re required level must teaching this task to him
                self._at_least_one_worker_with_level_greater_than_difficulty_of_task(m, x, penalty_levels, z_auxilary, job_completed_without_level)
                self.PENALTY_LEVEL = False
            

            if constraints_config.get("no_base_worker", False) :
                pass
            else :
                self._worker_of_the_first_operation_must_do_all_operations_of_the_job(m, x, y, job_done)

            if constraints_config.get("fix_value_skills_superior", False) :
                skills_value = constraints_config["fix_value_skills_superior"]
                self._fix_value_skills_superior(m, l, skills_value)
    
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
                self._constrained_makespan(m, C_max, limit_makespan, penalty_makespan, penalty_makespan_job, job_done_before_limit, C, job_done)
                print("constrained makespan activated with limit =", limit_makespan)
                print("constrained makespan activated with self.BORNE_SUP_MAKESPAN =", self.BORNE_SUP_MAKESPAN)
            
            if constraints_config.get("deadline_constraints_operation", False) :
                self._deadline_constraints_operation(m, x, d, z_auxilary, in_time, penalty_deadline, C, type="job") # type = "operation" ou "job" pour savoir si on applique la contrainte sur les opérations ou sur les jobs
            
            if constraints_config.get("hard_constraints_level_must_be_higher_if_solo", False) :
                self._hard_constraints_level_must_be_higher_if_solo(m, z_auxilary)

            # If need to init affectation variables for some values
            init_vars = constraints_config.get("init_affectaion_variables", False)
            if type(init_vars) is list :
                print("init affectation variables activated")
                self._init_affectaion_variables(x, init_vars[0], d, init_vars[1])
            else:
                print("init affectation variables desactivated")
            # else is bool False and not dict, so do nothing


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

        # if self.PENALTY_DEADLINE:
        #     penalty_deadline_obj = gp.quicksum(penalty_deadline[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
        # else:
        #     penalty_deadline_obj = 0

        # print("penalty deadline obj:", penalty_deadline_obj)

        # cognitive_load_tutors_obj = gp.quicksum(cognitive_load_tutors[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        # cognitive_load_collab_obj = gp.quicksum(cognitive_load_collaboration[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        



        ### DIFFERENT MODE D'OPTIM LE SKILLS
        if agregation_skills == "sum":
            skill_obj = gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) - gp.quicksum(self.instance.levels_workers[k][metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        
        elif agregation_skills == "max_min":
            for k in range(self.instance.nb_workers):
                for metier in range(self.instance.nb_professions):
                    m.addConstr((l[k,metier] - self.instance.levels_workers[k][metier] >= skill_obj), name=f"definition_z_max_min_{k}_{metier}")
        # elif agregation_skills == "max_min_line":
        # elif agregation_skills == "max_min_column":


        cognitive_load_total_obj = gp.quicksum(cognitive_load_total[k, metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions))
        # skill_obj = gp.quicksum(l[k,metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) - gp.quicksum(self.instance.levels_workers[k][metier] for k in range(self.instance.nb_workers) for metier in range(self.instance.nb_professions)) 

        # cognitive_load_total_obj += penalty_deadline_obj
        # skill_obj -= penalty_deadline_obj
        penalty_levels_obj = gp.quicksum(penalty_levels[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))

        # OBJECTIVES 
        # benefit_obj = gp.quicksum(self.instance.resale_price_jobs[i] * job_done[i] for i in range(self.instance.nb_jobs))
        # benefit_obj = gp.quicksum(self.instance.resale_price_jobs[i] * job_done[i] for i in range(self.instance.nb_jobs)) - penalty_levels_obj - penalty_makespan
        

        # Pénalité pour les jobs qui dépassent la durée limite
        penalty_makespan_job_OBJ = gp.quicksum(penalty_makespan_job[i] for i in range(self.instance.nb_jobs))

        # Job qui finissent à temps
        benefit_obj_in_time = gp.quicksum(self.instance.resale_price_jobs[i] * job_done_before_limit[i] for i in range(self.instance.nb_jobs)) 

        # Job qui finissent peu importe le temps qu'ils prennent
        benefit_obj = gp.quicksum(self.instance.resale_price_jobs[i] * job_done[i] for i in range(self.instance.nb_jobs))
        
        if benefit_in_time :
            obj_benefit = benefit_obj_in_time - penalty_levels_obj - penalty_makespan_job_OBJ - penalty_makespan*100
        else :
            obj_benefit = benefit_obj_in_time + benefit_obj - penalty_levels_obj - penalty_makespan_job_OBJ


        # if benefit_in_time:
        #     benefit_obj_in_time = gp.quicksum(self.instance.resale_price_jobs[i] * job_done_before_limit[i] for i in range(self.instance.nb_jobs))
            
            
        #     # + penalty_makespan pour l'annuler car si on considère les job finit à temps 
        #     # on uutilise uniquement la pénalité de chaque job au temp limite et pas une pénalité au dernier jobs terminé
        #     # mettre - penalty_makespan_job_OBJ peut paraitre problématique si le prix de revenu d'un job est inférieur à la penalité
        #     # il peut ne pas etre fait du tout alors que l'on souhaite qu'il soit quand même fait pour le bénéfice qu'il apporte même si il dépasse la date limite
        #     benefit_obj = (benefit_obj + penalty_makespan) + benefit_obj_in_time - penalty_makespan_job_OBJ

            


        # if objective == "makespan":
        #     m.setObjectiveN(penalty_levels_obj, index=0, priority=2, name="penalty_levels_obj")
        #     m.setObjectiveN(C_max, index=1, priority=1, name="makespan_obj")
        #     m.modelSense = GRB.MINIMIZE

        #     # m.setObjective(C_max + penalty_levels_obj * 10000, GRB.MINIMIZE)
        #     # m.setObjective(penalty_makespan + penalty_deadline_obj, GRB.MINIMIZE) # in minimizing the penalty for the tasks that exceed the current period, we minimize the makespan
        
        # ===============================================
        # ============== BENEFIT OBJECTIVE ==============
        # ===============================================

        if objective == "benefit" :
            if self.PENALTY_LEVEL:
                # print("self.penalty=", self.PENALTY_LEVEL)
                penalty_levels_obj = gp.quicksum(penalty_levels[i,j] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))
            else:
                penalty_levels_obj = 0
            # m.setObjectiveN(benefit_obj - penalty_levels_obj, index=0, priority=1, name="benefit_obj")
            # m.setObjectiveN(- penalty_makespan, index=1, priority=0, name="penalty_levels_obj")
            # m.modelSense = GRB.MAXIMIZE
            m.setObjective(obj_benefit, GRB.MAXIMIZE)

        # ===============================================
        # =========== MENTAL LOAD OBJECTIVE =============
        # ===============================================

        elif objective == "cognitive_load_total":
            m.setObjective(cognitive_load_total_obj + penalty_levels_obj *1000, GRB.MINIMIZE)


        # ===============================================
        # ============== SKILLS OBJECTIVE ===============
        # ===============================================

        # lorsque l'on optimise le skill, une fois optimisé on cherche à minimiser le makespan pour ne pas avoir de soultions abbérantes
        elif objective == "skill": 
            m.setObjectiveN(penalty_levels_obj, index=0, priority=2, name="penalty_levels_obj")
            m.setObjectiveN(- skill_obj, index=1, priority=1, name="maximize_skill_levels")
            m.setObjectiveN(C_max, index=2, priority=0, name="minimize_makespan")
            m.modelSense = GRB.MINIMIZE


        # ===============================================
        # =========== LEXICOGRAPHIC OBJECTIVE ===========
        # ===============================================

        # en priorité minimiser la difference entre le niveau du worker et les difficultés des taches qu'il fait seul : penalty_levels_obj
        elif objective == "lexicographic":
            if self.PENALTY_LEVEL:
                m.setObjectiveN(penalty_levels_obj, index=3, priority=3, name="penalty_levels_obj")
            # je considere le fait qu'une tache doit etre assigné a un worker qui a la niveau
            # m.setObjectiveN(penalty_levels_obj, index=0, priority=4, name="penalty_levels_obj")
            # m.setObjectiveN(- penalty_makespan, index=0, priority=3, name="penalty_makespan")
            m.setObjectiveN(obj_benefit , index=0, priority=priority[0], name="maximimize_benefit_obj")
            m.setObjectiveN(skill_obj, index=1, priority=priority[1], name="maximize_skill_levels_obj")
            m.setObjectiveN(-cognitive_load_total_obj, index=2, priority=priority[2], name="maximize_minus_cognitive_load_total_obj")
            m.modelSense = GRB.MAXIMIZE

        elif objective == "three":
            res = obj_benefit*weight[0] + skill_obj*weight[1] - cognitive_load_total_obj*weight[2]
            m.setObjective(res, GRB.MAXIMIZE)



        else:
            raise ValueError("objective doit être 'makespan', 'skill', 'three', 'lexicographic' ou 'cognitive_load_tutors'")
        



        # sum_Ci_obj = gp.quicksum(C[i] for i in range(self.instance.nb_jobs))

        # # minimsier somme des complétudes des jobs
        # m.setObjective(sum_Ci_obj, GRB.MINIMIZE)

        # # double objectif : minimiser le makespan et maximiser le niveau de compétence des travailleurs
        # m.setObjective(0.5 * C_max - 0.5 * skill_obj, GRB.MINIMIZE)

        m.write(f"../results/model_{objective}.lp")
        return m

    def solve(self, objective="makespan" , weight=[0,0,0], priority=[0,1,2], time_limit=None, constraints_config=None, verbose=False, job_with_no_skills=False, agregation_skills="sum", benefit_in_time=True) -> Solution:
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
        
        
        assert objective in ["makespan", "skill", "three", "lexicographic", "cognitive_load_total", "benefit"], "objective doit être 'makespan', 'skill', 'three', 'lexicographic', 'cognitive_load_total' ou 'benefit'"
        assert len(weight) == 3, "weight doit être une liste de trois éléments"
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
            agregation_skills=agregation_skills, 
            benefit_in_time=benefit_in_time
        )

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
        else :
            res.append(('Obj0', m.objVal))
            # print("iciiiiiiiiiiiiiii")
            # print("res", res)
        
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

        print('et la ->', self.BORNE_SUP_MAKESPAN)
        res.append(("BORNE_SUP_MAKESPAN", self.BORNE_SUP_MAKESPAN))
        
        if verbose :
            print("objective value:", m.objVal)
            print(res)

        s = Solution(self.instance)
        s.from_milp_var_list(res)
        return s    

    def _init_affectaion_variables(self, x, x_init, d, d_init):
        """ Depart des valeurs de la variable x pour la resolution aux valeurs de x_initial """
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                for k in range(self.instance.nb_workers):
                    x[i,j,k].start = x_init[i,j,k]
                    d[i,j,k].start = d_init[i,j,k]

        print("$*$*$*$*$*$ initial affectation variables set to x_initial $*$*$*$*$*$")


    def epsilon_contraintes(self, constraints_config=None, verbose=False, benefit_in_time=False):

        # résolution lexicogrpahique 
        solution_list = [] # liste des solutions obtenues pour chaque ordre d'optimisation lexicographique
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[2, 1, 0], verbose=verbose, benefit_in_time=benefit_in_time))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[2, 0, 1], verbose=verbose, benefit_in_time=benefit_in_time))

        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[1, 2, 0], verbose=verbose, benefit_in_time=benefit_in_time))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[0, 2, 1], verbose=verbose, benefit_in_time=benefit_in_time))

        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[0, 1, 2], verbose=verbose, benefit_in_time=benefit_in_time))
        solution_list.append(self.solve(objective="lexicographic", constraints_config=constraints_config, priority=[1, 0, 2], verbose=verbose, benefit_in_time=benefit_in_time))



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
                s = self.solve(objective="benefit", priority=[2, 1, 0], constraints_config=constraints_config_epsilon, verbose=verbose, benefit_in_time=benefit_in_time)
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


if __name__ == "__main__":
    
    # res = read_file("../data/data_temp.test")
    # res = read_file("../data/data_temp.test")
    res = read_file("../data/data_resale_price_1.test")
    
    instance = Instance()
    instance.from_dictionary(res)
    # instance.from_random(at_least_a_worker_have_competence_for_each_profession=False, seed=42)
    instance.qualified_workers_for_task(verbose=True)

    print(instance)
    model = Model(instance)
    
    # # # s = model.solve(objective="lexicographic", weight=[0.5, 0.5, 0.5], priority=[2, 1, 0], verbose=True)
    # # s = model.solve(objective="makespan", weight=[0.5, 0.5, 0.5], priority=[2, 1, 0], verbose=True)

    constraints_config = {}
    # constraints_config = {"job_with_no_skills": False,
    #                       "no_teaching_tasks": True
    #                       }
    s = model.solve(objective="benefit", weight=[0, 0, 0], priority=[2, 1, 0], constraints_config=constraints_config, verbose=True)
    # resume_levels_workers(s, instance)
    # # print(s)

    instance.qualified_workers_for_task(verbose=True)

    df = gantt_chart(s, instance, color=3, verbose=True, separate_little=True)
    print(df)

    res = check_df(df)
    print("check_df:", res)

    res = s.all_jobs_completed()
    print("all jobs completed:", res)

    res = s.wich_jobs_are_completed()
    print("completed jobs:", res)

    instance.qualified_workers_for_task(verbose=True)

    # # # plot_levels_workers(s, instance, verbose=True)



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
