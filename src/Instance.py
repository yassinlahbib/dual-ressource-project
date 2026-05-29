import numpy as np
from Constante import *


class Instance:
    def __init__(self):
        # VOIR si une maniere plus simple pour instancier l'instance
        pass

    def from_dictionary(self, dictionary: dict) -> None :
        # INT
        self.nb_jobs : int = dictionary["nb_jobs"] # int
        self.nb_professions : int = dictionary["nb_professions"] # int
        self.nb_task_in_profession : np.ndarray = dictionary["nb_task_in_profession"] # size (nb_professions)
        self.max_nb_operations : int = dictionary["max_nb_operations"] # int
        self.nb_tasks : int = dictionary["nb_tasks"] # int
        self.nb_workers : int = dictionary["nb_workers"] # int

        # DIFFICULTIES AND TIMES
        self.tasks_difficulties : np.ndarray = dictionary["tasks_difficulties"] # size (nb_tasks)
        self.tasks_times : np.ndarray = dictionary["tasks_times"] # size (nb_tasks, 3)

        # LEVELS OF WORKERS
        self.levels_workers : np.ndarray = dictionary["levels_workers"] # size (nb_workers, nb_professions)
        # Forgetting effect
        # self.forgetting = dictionary["forgetting_workers"] # size (nb_workers, nb_professions)
        
        # JOBS STRUCTURE
        self.jobs_struct : list = dictionary["jobs_struct"] # len=nb_jobs, len(jobs_struct[i]) = number of operations of job i.
        self.difficulty_jobs : np.ndarray = dictionary["difficulty_jobs"] # size (nb_jobs)
        self.resale_price_jobs : np.ndarray = dictionary["resale_price_jobs"]

        # CONSTRAINTS
        self.constraints_precedence_operations : np.ndarray = dictionary["constraints_precedence_operations"] # size(nb_jobs, max_nb_operations, max_nb_operations)
        
        # MAPPING TASK TO METIER
        self.task_to_m : dict = dictionary["dict_task_to_m"]


    def from_random(self,nb_jobs=5, nb_professions=5, nb_workers=5, max_nb_operations=4, at_least_a_worker_have_competence_for_each_profession=False, seed=42) -> None:
        
        np.random.seed(seed)
        
        self.nb_jobs : int = nb_jobs
        self.nb_professions : int = nb_professions
        self.nb_workers : int = nb_workers
        
        self.nb_task_in_profession : np.ndarray = np.random.randint(2, 4, size=self.nb_professions)
        self.nb_tasks : int = np.sum(self.nb_task_in_profession)

        self.max_nb_operations : int = min(max_nb_operations, self.nb_tasks)
        

        time_solo_tasks : np.ndarray = np.random.randint(1, 20, size=self.nb_tasks)
        time_teaching_tasks : np.ndarray = time_solo_tasks * np.random.uniform(1.2, 2.2, size=self.nb_tasks)
        time_collab_tasks : np.ndarray = time_solo_tasks * np.random.uniform(0.5, 1.0, size=self.nb_tasks)

        # certaines taches ne peuvent pas être fait en collab
        for i in range(self.nb_tasks):
            if np.random.rand() < 0.1:
                time_collab_tasks[i] = -1

        self.tasks_times : np.ndarray = np.stack((time_solo_tasks, time_teaching_tasks, time_collab_tasks), axis=1)
        self.tasks_difficulties : np.ndarray = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=self.nb_tasks)


        self.task_to_m : dict = {}
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

        self.levels_workers = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=(self.nb_workers, self.nb_professions))


        self.jobs_struct = []
        for i in range(self.nb_jobs):
            nb_operations = np.random.randint(2, self.max_nb_operations)
            operations = np.random.choice(self.nb_tasks, size=nb_operations, replace=True) #on peut avoir des taches qui se répètent dans le même job
            self.jobs_struct.append(operations)
        
        self.difficulty_jobs = np.array([max(self.tasks_difficulties[operations]) for operations in self.jobs_struct])
        self.resale_price_jobs = self.difficulty_jobs * np.random.uniform(10, 20, size=self.nb_jobs) # prix de revente des jobs proportionnel à leur difficulté


        self.constraints_precedence_operations = np.zeros((self.nb_jobs, self.max_nb_operations, self.max_nb_operations))
        for i in range(self.nb_jobs):
            for j in range(len(self.jobs_struct[i]) - 1):
                self.constraints_precedence_operations[i,j,j+1] = 1
                # for j_prime in range(j+1, len(self.jobs_struct[i])):


        # Pour pouvoir suivre le phénomène d'oublis des workers par corps de métier
        # self.forgetting = np.zeros((self.nb_workers, self.nb_professions)) # initialisé à 0 pour le départ


        self.resale_price_jobs = self.difficulty_jobs * np.random.uniform(5, 40, size=self.nb_jobs)

        ## A FAIRE
        # Certaines tache ne peuvent pas être fait avec un worker sans niveau requis - tache critique

        self.level_required = np.zeros

        if at_least_a_worker_have_competence_for_each_profession: # Au moins 1 worker à le niveau de compétence de la tache la plus difficile
            self.levels_workers = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=(self.nb_workers, self.nb_professions))
            for metier in range(self.nb_professions):
                if np.max(self.levels_workers[:, metier]) < max_difficulty_per_profession[metier]:
                    worker_to_change = np.random.randint(0, self.nb_workers)
                    self.levels_workers[worker_to_change, metier] = max_difficulty_per_profession[metier]


    def from_config(self, config) -> None:
            
            assert config["nb_professions"] == len(config["proportion_tasks_per_profession"]), "Le nombre de professions doit être égal à la longueur de la liste des proportions de taches par profession"
            assert config["nb_tasks"] in ["lower", "medium", "high"], "Le nombre de taches doit être une chaine de caractères 'lower', 'medium' ou 'high'"
            # assert sum(config["proportion_tasks_per_profession"]) == 1
            # # assert sum(config["proportion_jobs"]) == 1
            # assert sum(config["proportion_tasks_levels"]) == 1
            # assert sum(config["proportion_workers_levels"]) == 1
            

            np.random.seed(config["seed"])
            
            self.nb_jobs : int = config["nb_jobs"]
            self.nb_professions : int = config["nb_professions"]
            self.nb_workers : int = config["nb_workers"]
            self.max_nb_operations : int = config["max_nb_operations"]

            # NB TASKS
            if config["nb_tasks"] == "lower":
                self.nb_tasks : int = int(self.nb_jobs * self.max_nb_operations * 1.1)
            elif config["nb_tasks"] == "medium":
                self.nb_tasks : int = int(self.nb_jobs * self.max_nb_operations * 1.5)
            else:
                self.nb_tasks : int = int(self.nb_jobs * self.max_nb_operations * 2)

            # NB TASKS PER PROFESSION
            self.proportion_tasks_per_profession = np.array(config["proportion_tasks_per_profession"])
            self.nb_task_in_profession = (self.nb_tasks * self.proportion_tasks_per_profession).astype(int)

                
            ######################################################################
            ########################### TIME OF TASKS ############################
            ######################################################################
            task_per_times = [ set() for _ in range(3)] # liste de set pour stocker les indices des taches de chaque type de job (small, medium, long)

            self.proportion_tasks_times = np.array(config["proportion_tasks_times"])
            proportion_tasks_times_cum_sum = np.cumsum(self.proportion_tasks_times)
            time_solo_tasks : np.ndarray = np.zeros(self.nb_tasks)

            for i in range(self.nb_tasks):
                x = np.random.rand()
                
                # small tasks
                if x <= proportion_tasks_times_cum_sum[0]:
                    time_solo_tasks[i] = np.random.uniform(1, 3+1) # tache courte entre 1 et 3 unités de temps (10 à 30 min)
                    task_per_times[0].add(i)
                
                # medium tasks
                elif x <= proportion_tasks_times_cum_sum[1]:
                    time_solo_tasks[i] = np.random.uniform(3, 6+1) # tache moyenne entre 3 et 18 unités de temps (30 min à 1h)
                    task_per_times[1].add(i)
                # long tasks
                else :
                    time_solo_tasks[i] = np.random.uniform(6, 12+1) # tache longue entre 6 et 12 unités de temps (1h à 2h)
                    task_per_times[2].add(i)


                time_teaching_tasks : np.ndarray = time_solo_tasks * np.random.uniform(1.4, 2.2, size=self.nb_tasks)
                time_collab_tasks : np.ndarray = time_solo_tasks * np.random.uniform(0.4, 0.8, size=self.nb_tasks)


            # TASK CAN NOT DONE IN COLLABORATION
            self.proportion_tasks_without_collaborative_work = config["proportion_tasks_without_collaborative_work"]
            # certaines taches ne peuvent pas être fait en collab
            for i in range(self.nb_tasks):
                if np.random.rand() < self.proportion_tasks_without_collaborative_work:
                    time_collab_tasks[i] = -1

            self.tasks_times : np.ndarray = np.stack((time_solo_tasks, time_teaching_tasks, time_collab_tasks), axis=1)


            ######################################################################
            ########################### LEVEL OF TASKS ###########################
            ######################################################################
            self.proportion_tasks_difficulties = np.array(config["proportion_tasks_levels"])
            proportion_tasks_difficulties_cum_sum = np.cumsum(self.proportion_tasks_difficulties)

            self.tasks_difficulties : np.ndarray = np.zeros(self.nb_tasks)

            for i in range(self.nb_tasks):
                x = np.random.rand()

                # level 1
                if x <= proportion_tasks_difficulties_cum_sum[0]:
                    self.tasks_difficulties[i] = 1

                # level 2
                elif x <= proportion_tasks_difficulties_cum_sum[1]:
                    self.tasks_difficulties[i] = 2

                # level 3
                elif x <= proportion_tasks_difficulties_cum_sum[2]:
                    self.tasks_difficulties[i] = 3

                # level 4
                else:
                    self.tasks_difficulties[i] = 4

            ######################################################################
            ######################### TASKS BY PROFESSION ########################
            ######################################################################
            self.task_to_m : dict = {}
            size_curr = 0 # beacause self.nb_task_in_profession is an array of int 
            id_task = np.arange(np.sum(self.nb_task_in_profession))
            for m in range(self.nb_professions):
                for t in range(self.nb_task_in_profession[m]):
                    size_curr += 1
                    self.task_to_m[id_task[0]] = m
                    id_task = id_task[1:]
            
            # assert size_curr == self.nb_tasks or size_curr == self.nb_tasks - 1, f"Le nombre de taches {self.nb_tasks} doit être égal à la somme du nombre de taches par profession ou à la somme du nombre de taches par profession - 1 {size_curr} (si on arrondi à l'entier inférieur pour certaines professions)"

            for i in range(size_curr, self.nb_tasks):
                for j in range(self.nb_professions):
                    if self.proportion_tasks_per_profession[j] != 0:
                        self.task_to_m[i] = j
                        break

            # print("task_to_m: ", self.task_to_m)
            # print("nb_tasks: ", self.nb_tasks)
            # print("size_curr: ", size_curr)
        
            

            # par metier il faut au moins un worker avec la compétence nécessaire pour faire les taches de ce metier sinon pas de solution pour l'instance (en attendant de faire soft contraintes sur cela)
            max_difficulty_per_profession = np.zeros(self.nb_professions)

            for metier in range(self.nb_professions):
                max_diff = 0
                for t in range(self.nb_tasks):
                    if self.task_to_m[t] == metier:
                        if self.tasks_difficulties[t] > max_diff:
                            max_diff = self.tasks_difficulties[t]
                max_difficulty_per_profession[metier] = max_diff

            ######################################################################
            ######################### LEVEL OF WORKERS ###########################
            ######################################################################

            self.levels_workers : np.ndarray = np.zeros((self.nb_workers, self.nb_professions))
            
            proportion_workers_levels = config["proportion_workers_levels"]
            proportion_workers_levels_cum_sum = np.cumsum(proportion_workers_levels)

            for i in range(self.nb_workers):
                for j in range(self.nb_professions):
                    x = np.random.rand()

                    # level 1
                    if x <= proportion_workers_levels_cum_sum[0]:
                        self.levels_workers[i][j] = np.random.uniform(LEVEL_MIN, 2)

                    # level 2
                    elif x <= proportion_workers_levels_cum_sum[1]:
                        self.levels_workers[i][j] = np.random.uniform(2, 3)

                    # level 3
                    elif x <= proportion_workers_levels_cum_sum[2]:
                        self.levels_workers[i][j] = np.random.uniform(3, 4)

                    # level 4
                    else:
                        self.levels_workers[i][j] = LEVEL_MAX



            self.jobs_struct = []
            for i in range(self.nb_jobs):

                nb_operations = np.random.randint(2, self.max_nb_operations)
                operations = np.random.choice(self.nb_tasks, size=nb_operations, replace=True) #on peut avoir des taches qui se répètent dans le même job
                self.jobs_struct.append(operations)
            
            self.difficulty_jobs = np.array([max(self.tasks_difficulties[operations]) for operations in self.jobs_struct])
            self.resale_price_jobs = self.difficulty_jobs * np.random.uniform(10, 20, size=self.nb_jobs) # prix de revente des jobs proportionnel à leur difficulté


            self.constraints_precedence_operations = np.zeros((self.nb_jobs, self.max_nb_operations, self.max_nb_operations))
            for i in range(self.nb_jobs):
                for j in range(len(self.jobs_struct[i]) - 1):
                    self.constraints_precedence_operations[i,j,j+1] = 1
                    # for j_prime in range(j+1, len(self.jobs_struct[i])):


            # Pour pouvoir suivre le phénomène d'oublis des workers par corps de métier
            # self.forgetting = np.zeros((self.nb_workers, self.nb_professions)) # initialisé à 0 pour le départ


            self.resale_price_jobs = self.difficulty_jobs * np.random.uniform(5, 40, size=self.nb_jobs)

            # print("-----------------------------------------")
            # print("-----------------------------------------")

            # print(f"nb_jobs: {self.nb_jobs}")
            # print(f"nb_tasks: {self.nb_tasks}")
            # print(f"nb_professions: {self.nb_professions}")
            # print(f"nb_workers: {self.nb_workers}")
            # print(f"max_nb_operations: {self.max_nb_operations}")
            # print(f"nb_task_in_profession: {self.nb_task_in_profession}")
            # print("tasks_times: ")
            # print(self.tasks_times)
            # print("tasks_difficulties: ")
            # print(self.tasks_difficulties)
            # print("task_to_m: ")
            # print(self.task_to_m)
            # print("levels_workers: ")
            # print(self.levels_workers)
            # print("jobs_struct: ")
            # print(self.jobs_struct)
            # print("resale_price_jobs: ")
            # print(self.resale_price_jobs)
            # print("difficulty_jobs: ")
            # print(self.difficulty_jobs)

            # print("-----------------------------------------")
            # print("-----------------------------------------")
            # print("-----------------------------------------")
            print("set")
            print(task_per_times)

            ## A FAIRE
            # Certaines tache ne peuvent pas être fait avec un worker sans niveau requis - tache critique

            # self.level_required = np.zeros

            if config["at_least_a_worker_have_competence_for_each_profession"]: # Au moins 1 worker à le niveau de compétence de la tache la plus difficile
                # self.levels_workers = np.random.randint(LEVEL_MIN, LEVEL_MAX + 1, size=(self.nb_workers, self.nb_professions))
                for metier in range(self.nb_professions):
                    if np.max(self.levels_workers[:, metier]) < max_difficulty_per_profession[metier]:
                        worker_to_change = np.random.randint(0, self.nb_workers)
                        self.levels_workers[worker_to_change, metier] = max_difficulty_per_profession[metier]

    
    # 1 unité = 10 min = 0.166 h
    # 42 unité = 420 min = 7h



    def qualified_workers_for_task(self, verbose=False) :
        """
        Retourne
        - res_qualified : liste des workers qualifiés pour chaque tache
        - res_at_most_one : liste non qualifié ayant une différence d'au plus 1 avec la difficulté de la tache
        """
        # fonction pour visualisation les indices commence à 1  pour les workers
        res_qualified  = []
        res_at_most_one  = []
        for i in range(len(self.jobs_struct)):
            res_qualified.append([])
            res_at_most_one.append([])
            for j in range(len(self.jobs_struct[i])):
                res_qualified[i].append([])
                res_at_most_one[i].append([])
        for i in range(len(self.jobs_struct)):
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                level_task = self.tasks_difficulties[index_task] # niveau de diificulté de la tache 
                m = self.task_to_m[index_task]
                # print(f"tache {index_task} - niveau {level_task} - metier {m}")
                for k in range(self.nb_workers):
                    # print(f"\tworker {k+1} - level {self.levels_workers[k][m]}")
                    
                    if self.levels_workers[k][m] >= level_task:
                        res_qualified[i][j].append(k+1) # k+1 pour visualisation on commence les worker par w_1

                    elif self.levels_workers[k][m] + 1 >= level_task : # worker non qualifié mais ayant un niveau d'au plus 1 de différence avec la difficulté de la tache
                        res_at_most_one[i][j].append(k+1) # k+1 pour visualisation on commence les worker par w_1

        if verbose == True:
            
            print(" ========== Qualified workers for each task: ==========")
            for i in range(len(self.jobs_struct)):
                print(f"J{i+1}: ",end="")
                for j in range(len(self.jobs_struct[i])):
                    print(f" ({i+1},{j+1}): {res_qualified[i][j]} \t", end="")
                print("\n")
                    
        # print("FUNC")
        # print("res_qualified=", res_qualified)
        # print("res_at_most_one=", res_at_most_one)
        return (res_qualified, res_at_most_one)

    def sum_of_job_alone(self):
        res = 0
        for i in range(len(self.jobs_struct)):
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                res += self.tasks_times[index_task][0]
        return  res
    
    def mean_time_job_alone(self):
        return self.sum_of_job_alone() / self.nb_jobs
    
    def mean_time_operations(self):
        total_operations = 0
        for i in range(len(self.jobs_struct)):
            total_operations += len(self.jobs_struct[i])
        return total_operations / self.nb_jobs
        
    def feasible_jobs(self, verbose=False):
        res = [ i for i in range(self.nb_jobs)]
        qualified_worker = self.qualified_workers_for_task(verbose=False)
        for i in range(len(qualified_worker)):
            for j in range(len(qualified_worker[i])):
                if len(qualified_worker[i][j]) == 0:
                    res.remove(i)
                    break
        if verbose == True:
            print(" ========== Feasible jobs with required (index) : ==========")
            if len(res) == 0:
                print("No feasible job")
            else:
                print(res)
        return res

    def time_each_job_alone(self, verbose=False):
        res = []
        for i in range(len(self.jobs_struct)):
            time_job = 0
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                time_job += self.tasks_times[index_task][0]
            if verbose == True:
                print(f"Time of job {i+1} alone: {time_job:.2f}") 
            res.append(time_job)
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
               f"Job resale prices: shape= {self.resale_price_jobs.shape}\n{self.resale_price_jobs}\n"
               f"Task difficulties: shape= {self.tasks_difficulties.shape}\n{self.tasks_difficulties}\n"
               f"Task processing times: shape= {self.tasks_times.shape}\n{self.tasks_times}\n")
        
        res += (f"Jobs structure: len= {len(self.jobs_struct)}\n{jobs_struct_str}\n"
            #    f"Precedence constraints: shape= {self.constraints_precedence_operations.shape}\n{self.constraints_precedence_operations}\n"
               f"\n ===== End of Instance: =====\n")
        
        return res


