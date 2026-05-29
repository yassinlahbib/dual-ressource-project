import numpy as np
from Instance import *
from Constante import *



class Solution:
    def __init__(self, instance):

        self.instance = instance

        # Les matrices suivantes possèdent beaucoup de zéros car elles sont de la taille maximale.
        self.x = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers)) # x[i, j, k] = 1 if operation j of job i is assigned to worker k, 0 otherwise
        self.d = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers))
        self.C = np.zeros(instance.nb_jobs)
        self.C_max = 0
        self.delta = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_jobs, instance.max_nb_operations, instance.nb_workers))
        
        # benefit variable
        self.job_done = np.zeros(instance.nb_jobs) # job_done[i] = 1 if job i is completed, 0 otherwise
        self.penalty_makespan_job = np.zeros(instance.nb_jobs) # penalty_makespan_job[i] = pénalité pour le job i si il est terminé après la date limite, 0 sinon
        self.job_done_before_limit = np.zeros(instance.nb_jobs) # job_done_before_limit[i] = 1 if job i is completed before limit time, 0 otherwise

        # know base worker must do all operation of the current job
        self.y = np.zeros((instance.nb_jobs, instance.nb_workers))


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
        self.penalty_makespan = 0
        self.penalty_deadline = np.zeros((instance.nb_jobs, instance.max_nb_operations))
        self.borne_sup_makespan = -1 # Si pas de borne sup makespan contraint alors mettre à -1

        # objective values
        self.objective_values = {}

    def from_genetic_algorithm(self, sorted_operations, first_worker_assignment, second_worker_assignment, skills, cognitive_load, m):
        for iter in range(len(sorted_operations)):
            i, j = m.id_to_op[sorted_operations[iter]]
            w1 = first_worker_assignment[iter]
            w2 = second_worker_assignment[iter]

            # x_ijk
            if w1 != -1: # si w1 fait O_ij
                self.x[i, j, w1] = 1
            if w2 >= 0: # si w2 fait O_ij
                self.x[i, j, w2] = 1

    def from_milp_var_list(self, var_list):
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

            elif v[0][:6] == "delta[" : # delta[i, j, h, g, k] -> indices = [i, j, h, g, k]
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

            elif v[0][:9] == "job_done[":
                indices = v[0][9:-1].split(",")
                i = int(indices[0])
                self.job_done[i] = v[1]

            elif v[0] == "penalty_makespan":
                indices = v[0][17:-1].split(",")
                self.penalty_makespan = v[1]

            elif v[0][:16] == "penalty_deadline":
                indices = v[0][17:-1].split(",")
                i, j = int(indices[0]), int(indices[1])
                self.penalty_deadline[i, j] = v[1]

            elif v[0][:2] == "y[":
                indices = v[0][2:-1].split(",")
                i, k = int(indices[0]), int(indices[1])
                self.y[i, k] = v[1]
            
            elif v[0] == "BORNE_SUP_MAKESPAN":
                self.borne_sup_makespan = v[1]

            elif v[0][:21] == "job_done_before_limit":
                indices = v[0][22:-1].split(",")
                i = int(indices[0])
                self.job_done_before_limit[i] = v[1]

            elif v[0][:21] == "penalty_makespan_job[":
                indices = v[0][21:-1].split(",")
                i = int(indices[0])
                self.penalty_makespan_job[i] = v[1]

    def all_jobs_completed(self, verbose=False):
        # vérifie que toutes les opérations de tous les jobs ont été assignées
        print(" ========== Checking if all jobs are completed: ==========")
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                if np.sum(self.x[i, j, :]) == 0:
                    if verbose:
                        print(f"Job {i+1} is not completed yet.")
                    return False
        if verbose:
            print("All jobs are completed.")
        return True

    def job_is_completed(self, i):
        assert i < self.instance.nb_jobs and i >= 0, "Index of job must be between 0 and nb_jobs-1"
        for j in range(len(self.instance.jobs_struct[i])):
            if np.sum(self.x[i, j, :]) == 0:
                return False
        return True

    def which_jobs_are_completed(self, verbose=False):
        res = []
        for i in range(self.instance.nb_jobs):
            if self.job_is_completed(i):
                res.append(i)
        if verbose:
            print(" ========== Completed jobs: ==========")
            if len(res) == 0:
                print("No job is completed")
            else:
                print(res)
        return res

    def resale_price_job_done_before_limit(self):
        res = 0
        for i in range(self.instance.nb_jobs):
            if self.job_done_before_limit[i] == 1:
                res += self.instance.resale_price_jobs[i]
        return res
    
    def resale_price_job_done(self):
        res = 0
        for i in range(self.instance.nb_jobs):
            if self.job_done[i] == 1:
                res += self.instance.resale_price_jobs[i]
        return res

    def sum_cognitive_load_total(self):
        return np.sum(self.cognitive_load_total)
    
    def sum_skill_levels_rate(self):
        return np.sum(self.l) - np.sum(self.instance.levels_workers)

    # fonction __str__ pas à jours 
    def __str__(self):

        res = (f"x: {self.x.shape} \n{self.x}\n"
               f"d: {self.d.shape} \n{self.d}\n"
               f"C: {self.C.shape} \n{self.C}\n"
               f"C_max: {self.C_max}\n"
               f"job_done: {self.job_done.shape} \n{self.job_done}\n"
               f"f: {self.f.shape} \n{self.f}\n"
               f"z_auxilary: {self.z_auxilary.shape} \n{self.z_auxilary}\n"
               f"is_tutor: {self.is_tutor.shape} \n{self.is_tutor}\n"
               f"cognitive_load_tutors: {self.cognitive_load_tutors.shape} \n{self.cognitive_load_tutors}\n"
               f"cognitive_load_apprentis: {self.cognitive_load_apprentis.shape} \n{self.cognitive_load_apprentis}\n"
               f"cognitive_load_collaboration: {self.cognitive_load_collaboration.shape} \n{self.cognitive_load_collaboration}\n"
               f"cognitive_load_total: {self.cognitive_load_total.shape} \n{self.cognitive_load_total}\n"
               f"penalty_levels: {self.penalty_levels.shape} \n{self.penalty_levels}\n"
               f"penalty_makespan: {self.penalty_makespan}\n"
               f"penalty_deadline: {self.penalty_deadline.shape} \n{self.penalty_deadline}\n"
               f"objective_values: {self.objective_values}\n"
               f"l: {self.l.shape} \n{self.l}\n"
               f"y: {self.y.shape} \n{self.y}\n"
               )
        
        return res
    


