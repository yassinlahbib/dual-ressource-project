import numpy as np
from Constante import *

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap



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


    def from_scenario(self, nb_jobs=6, nb_workers=5, nb_professions=3,
                      nb_tasks_per_profession=3, max_nb_operations=4,
                      worker_profile="balanced", schedule_tightness=0.8, seed=42):
        """
        Génère une instance réaliste et contrôlable via des paramètres métier.

        Args:
            nb_jobs            : nombre de jobs à ordonnancer
            nb_workers         : nombre de workers disponibles
            nb_professions     : nombre de métiers distincts
            max_nb_operations  : nombre max d'opérations par job
            worker_profile     : distribution des niveaux workers
                                 "junior"   → peu qualifiés, beaucoup de teaching possible
                                 "balanced" → mélange qualifiés / apprentis
                                 "senior"   → majoritairement qualifiés
            nb_tasks_per_profession : nombre de tâches par métier, chacune avec une
                                     difficulté tirée aléatoirement entre 1 et LEVEL_MAX
            schedule_tightness : ratio fenêtre de temps / temps total en solo
                                 < 1 → impossible de tout finir (sélection forcée)
                                 1.0 → juste le temps en solo pur
                                 > 1 → confortable
            seed               : pour la reproductibilité
        """
        np.random.seed(seed)

        self.nb_jobs         = nb_jobs
        self.nb_workers      = nb_workers
        self.nb_professions  = nb_professions
        self.max_nb_operations = max_nb_operations

        # ── 1. TÂCHES ───────────────────────────────────────────────────────
        self.nb_task_in_profession = np.full(nb_professions, nb_tasks_per_profession)
        self.nb_tasks = nb_professions * nb_tasks_per_profession

        # tâche t appartient au métier t // nb_tasks_per_profession
        self.task_to_m = {t: t // nb_tasks_per_profession for t in range(self.nb_tasks)}

        # difficultés aléatoires entre 1 et LEVEL_MAX pour chaque tâche
        self.tasks_difficulties = np.random.randint(LEVEL_MIN, LEVEL_MAX, size=self.nb_tasks).astype(float)

        # temps solo proportionnel à la difficulté, teaching plus long, collab plus court
        solo_times     = self.tasks_difficulties * np.random.uniform(2.0, 4.0, size=self.nb_tasks)
        teaching_times = solo_times * np.random.uniform(1.3, 1.8, size=self.nb_tasks)
        collab_times   = solo_times * np.random.uniform(0.5, 0.75, size=self.nb_tasks)

        # 20% des tâches impossibles en collaboration (ex : diagnostic individuel)
        collab_times[np.random.rand(self.nb_tasks) < 0.2] = -1

        self.tasks_times = np.stack([solo_times, teaching_times, collab_times], axis=1)

        # ── 2. WORKERS ──────────────────────────────────────────────────────
        # Chaque worker a 1 métier "principal" (niveau élevé) et les autres au niveau de base.
        # Le métier principal tourne sur les workers : w0 → m0, w1 → m1, etc.
        base_level    = {"junior": 1, "balanced": 2, "senior": 3}[worker_profile]
        primary_level = {"junior": 2, "balanced": 3, "senior": 4}[worker_profile]

        self.levels_workers = np.full((nb_workers, nb_professions), float(base_level))
        for k in range(nb_workers):
            self.levels_workers[k, k % nb_professions] = float(primary_level)

        # garantir qu'au moins un worker est qualifié pour la tâche la plus dure de chaque métier
        # (difficulté max = 3) pour que l'instance soit toujours faisable
        for m in range(nb_professions):
            if np.max(self.levels_workers[:, m]) < 3:
                self.levels_workers[np.random.randint(nb_workers), m] = 3.0

        # ── 3. JOBS ─────────────────────────────────────────────────────────
        # Chaque job traverse des métiers différents (sans répétition de métier)
        # Réaliste : un produit passe par différentes étapes de réparation
        self.jobs_struct = []
        for i in range(nb_jobs):
            nb_ops = np.random.randint(2, max_nb_operations + 1)
            nb_ops = min(nb_ops, nb_professions)  # au plus 1 tâche par métier
            chosen_professions = np.random.choice(nb_professions, size=nb_ops, replace=False)
            # pour chaque métier choisi, prendre une tâche au hasard parmi ses 3
            ops = np.array([
                m * nb_tasks_per_profession + np.random.randint(nb_tasks_per_profession)
                for m in chosen_professions
            ])
            self.jobs_struct.append(ops)

        self.difficulty_jobs   = np.array([max(self.tasks_difficulties[ops]) for ops in self.jobs_struct])
        self.resale_price_jobs = self.difficulty_jobs * np.random.uniform(10, 30, size=nb_jobs)

        # ── 4. PRÉCÉDENCE ───────────────────────────────────────────────────
        # Contrainte chaîne simple : op j doit être faite avant op j+1
        self.constraints_precedence_operations = np.zeros(
            (nb_jobs, max_nb_operations, max_nb_operations)
        )
        for i in range(nb_jobs):
            for j in range(len(self.jobs_struct[i]) - 1):
                self.constraints_precedence_operations[i, j, j + 1] = 1

        # ── 5. FENÊTRE DE TEMPS ─────────────────────────────────────────────
        # Calculée à partir des temps réels de l'instance pour rester cohérente.
        # schedule_tightness = 0.8 → 80% du temps nécessaire pour tout faire en solo
        total_solo = self.sum_of_job_alone()
        self.time_window = round(total_solo * schedule_tightness)

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
            
    def sum_profit_of_jobs(self, verbose=False):
        res = 0
        for i in range(len(self.jobs_struct)):
            res += self.resale_price_jobs[i]
            if verbose == True:
                print(f"Profit of job {i+1}: {self.resale_price_jobs[i]:.2f}") 
        if verbose == True:
            print(f"Total profit of jobs: {res:.2f}") 
        return res

    def view_difficulty_and_metier_of_each_task(self, verbose=False):
        res = []
        for i in range(len(self.jobs_struct)):
            res.append([])
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                level_task = self.tasks_difficulties[index_task] # niveau de diificulté de la tache 
                m = self.task_to_m[index_task]
                res[i].append((level_task, m))
                if verbose == True:
                    print(f"Task ({i+1},{j+1}) - index {index_task} - level {level_task} - metier {m}")
        return res            

    # GENERE
    def vizualize_levels_workers(self):

        M = self.levels_workers  # matrice nb_workers × nb_metiers

        # Palette continue : rouge -> orange -> jaune -> vert
        cmap = LinearSegmentedColormap.from_list(
            "red_to_green",
            ["red", "orange", "yellow", "green"]
        )

        plt.figure(figsize=(12, 6))


        plt.imshow(M, cmap=cmap, vmin=1, vmax=4, aspect="auto", interpolation="nearest")

        cbar = plt.colorbar()
        cbar.set_label("Niveau")
        cbar.set_ticks([1, 2, 3, 4])

        plt.xlabel("Métiers")
        plt.ylabel("Workers")
        plt.title("Matrice workers × métiers")

        plt.xticks(
            range(M.shape[1]),
            [f"Métier {j}" for j in range(M.shape[1])],
            rotation=45
        )

        plt.yticks(
            range(M.shape[0]),
            [f"W{i}" for i in range(M.shape[0])]
        )

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color="black")

        plt.tight_layout()
        plt.show()




    # GENERE
    def vizualize_differences_levels_workers_job_i(self, job_index):
        operations = self.jobs_struct[job_index]
        nb_operations = len(operations)

        res = np.zeros((self.nb_workers, nb_operations))

        x_labels = []

        for j in range(nb_operations):
            index_task = operations[j]

            level_task = self.tasks_difficulties[index_task]
            m = self.task_to_m[index_task]

            res[:, j] = self.levels_workers[:, m] - level_task

            # Label affiché sur l'axe X
            # x_labels.append(f"Op {index_task}\nM {m}")
            x_labels.append(f"Métier {m}\nLevel {level_task}")

        # Palette continue rouge -> jaune -> vert
        cmap = LinearSegmentedColormap.from_list(
            "diff_red_yellow_green",
            ["red", "yellow", "green"]
        )

        # Centre la couleur jaune autour de 0
        max_abs = np.max(np.abs(res))

        norm = TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs
        )

        plt.figure(figsize=(max(10, nb_operations * 1.2), max(6, self.nb_workers * 0.4)))

        plt.imshow(
            res,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest"
        )

        cbar = plt.colorbar()
        cbar.set_label("Niveau worker - difficulté opération")

        plt.xlabel("Opérations du job avec métier associé")
        plt.ylabel("Workers")
        plt.title(f"Différence de niveau workers / opérations - Job {job_index}")

        plt.xticks(
            range(nb_operations),
            x_labels,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(self.nb_workers),
            [f"W{i}" for i in range(self.nb_workers)]
        )

        # Affichage des valeurs avec 1 chiffre après la virgule
        for i in range(self.nb_workers):
            for j in range(nb_operations):
                plt.text(
                    j,
                    i,
                    f"{res[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black"
                )

        plt.tight_layout()
        plt.show()

        return res

    def visualize_jobs_overview(self, render="html", save_path=None):
        """
        Tableau Plotly interactif : 1 bloc de 5 lignes par job, 1 colonne par opération.
        Bandes de couleur par type d'info :
          Bleu   — métier + difficulté
          Gris   — temps solo / apprentissage / collab
          Vert   — workers qualifiés   (Lv >= Df)
          Orange — workers apprentis   (Lv == Df-1)
          Rouge  — workers non qualifiés (Lv < Df-1)

        Args:
            render    : "html" | "interactif" | "notebook"
            save_path : chemin HTML optionnel
        """
        import plotly.graph_objects as go

        res_qualified, res_at_most_one = self.qualified_workers_for_task()
        max_ops     = max(len(self.jobs_struct[i]) for i in range(self.nb_jobs))
        all_workers = set(range(1, self.nb_workers + 1))

        BAND_COLORS  = ["#D6EAF8", "#F4F6F7", "#D5F5E3", "#FDEBD0", "#FADBD8"]
        JOB_COLORS   = ["#EBF5FB", "#F2F3F4"]   # alternance par job

        def _fmt(lst):
            return '  '.join(f'w{k}' for k in lst) if lst else '–'

        n_rows   = self.nb_jobs * 5
        col_job  = [""] * n_rows
        col_vals = [[""] * n_rows for _ in range(max_ops)]
        col_fill = [["white"] * n_rows for _ in range(max_ops + 1)]

        for i in range(self.nb_jobs):
            base      = i * 5
            job_color = JOB_COLORS[i % 2]

            col_job[base] = f"<b>J{i+1}</b>"
            for band in range(5):
                col_fill[0][base + band] = job_color

            for j in range(max_ops):
                if j >= len(self.jobs_struct[i]):
                    for band in range(5):
                        col_fill[j + 1][base + band] = "#FDFEFE"
                    continue

                index_task  = self.jobs_struct[i][j]
                level_task  = int(self.tasks_difficulties[index_task])
                m           = int(self.task_to_m[index_task])
                t_solo      = self.tasks_times[index_task][0]
                t_app       = self.tasks_times[index_task][1]
                t_co        = self.tasks_times[index_task][2]
                t_co_str    = f"{t_co:.1f}" if t_co >= 0 else "N/A"

                qualified   = res_qualified[i][j]
                apprenti    = res_at_most_one[i][j]
                unqualified = sorted(all_workers - set(qualified) - set(apprenti))

                texts = [
                    f"<b>m{m+1}  |  Df={level_task}</b>",
                    f"A:{t_solo:.1f}   L:{t_app:.1f}   C:{t_co_str}",
                    f"✓  {_fmt(qualified)}",
                    f"≈  {_fmt(apprenti)}",
                    f"✗  {_fmt(unqualified)}",
                ]

                for band in range(5):
                    col_vals[j][base + band]       = texts[band]
                    col_fill[j + 1][base + band]   = BAND_COLORS[band]

        fig = go.Figure(go.Table(
            columnwidth=[50] + [130] * max_ops,
            header=dict(
                values=["<b>Job</b>"] + [f"<b>Op {j+1}</b>" for j in range(max_ops)],
                fill_color="#2C3E50",
                font=dict(color="white", size=11),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[col_job] + col_vals,
                fill_color=col_fill,
                align="center",
                font=dict(size=10),
                height=26,
            ),
        ))

        fig.update_layout(
            title=dict(
                text="Vue d'ensemble des Jobs — Opérations, Métiers, Temps & Workers",
                font=dict(size=13), x=0.5,
            ),
            height=max(400, n_rows * 28 + 120),
            margin=dict(t=60, b=20, l=20, r=20),
        )

        if render == "html":
            fig.write_html("../results/jobs_overview.html", auto_open=True)
        elif render == "interactif":
            fig.show()
        elif render == "notebook":
            fig.show("png")

        if save_path:
            fig.write_html(save_path)

        return fig

    def get_difficulty_and_metier_of_task(self, i, j):
        index_task = self.jobs_struct[i][j]
        level_task = self.tasks_difficulties[index_task] # niveau de diificulté de la tache 
        m = self.task_to_m[index_task]
        return (level_task, m)
        

    def __str__(self):

        jobs_struct_str = ""
        for i in range(self.nb_jobs):
            jobs_struct_str += f"Job {i} : "
            for j in range(len(self.jobs_struct[i])):
                index_task = self.jobs_struct[i][j]
                jobs_struct_str += f"\tO_({i},{j}) = {self.jobs_struct[i][j]} ({self.tasks_times[index_task][0]:.2f}) "
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


    