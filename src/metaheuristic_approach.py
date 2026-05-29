from Instance import *
from Solution import *
from Constante import *
from copy import copy, deepcopy


class Encoding_Dicos:
    """ Permet de faire le lien entre l'indice de l'operation dans le chromosome et l'operation (i,j) de l'instance et inversment """
    def __init__(self, instance):
        self.instance = instance
        self.id_to_op = self._id_to_operation()
        self.op_to_id = self._operation_to_id()
    
    def _id_to_operation(self):
        res = {}
        id = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                res[id] = (i,j) # we assign the id to the operation (i,j) where i is the job and j is the index of the operation in this job
                id += 1
        return res

    def _operation_to_id(self):
        res = {}
        id = 0
        for i in range(self.instance.nb_jobs):
            for j in range(len(self.instance.jobs_struct[i])):
                res[(i,j)] = id # we assign the id to the operation (i,j) where i is the job and j is the index of the operation in this job
                id += 1
        return res


class ChromosomeGenerator:
    """
    IMPORTANT: 
        Differencier task_id et op_id
        - task_id est l'id de la task donée par l'instance, c'est un entier entre 0 et nb_tasks-1 (car une task peut être dans plusieurs jobs)
        - op_id est l'id de l'opération, c'est un entier entre 0 et nb_operations-1 pour identifier une opération de manière unique
    """
    def __init__(self, instance: Instance, dicos: Encoding_Dicos, constraints_config=None):
        self.instance : Instance = instance
        self.dicos : Encoding_Dicos = dicos
        self.constraints_config = constraints_config # pour savoir si solo sans level accepté ou non
        self.nb_op_in_job = np.array([len(self.instance.jobs_struct[i]) for i in range(self.instance.nb_jobs)])
        self.nb_op = np.sum(self.nb_op_in_job) # number of operations to schedule
        # print("id_to_op=", self.id_to_op)
        # print("op_to_id=", self.op_to_id)
    
    def init_operations_order(self):
        """
        Retourne un vecteur de taille nb_operations, qui contient l'id des jobs dans un ordre topologique
        """
        operations_order = []        
        for i in range(len(self.nb_op_in_job)):
            operations_order.extend([i] * self.nb_op_in_job[i])
        
        operations_order = np.array(operations_order)
        np.random.shuffle(operations_order)

        return operations_order
    
    def init_first_worker_assignment(self, operations_order):
        """
        Retourne un vecteur de taille nb_operations, qui contient l'id du worker assigné à chaque opération
        """
        worker_assignment = np.zeros(self.nb_op, dtype=int)
        for idx_op in range(len(operations_order)):


            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            # print("operation=", (i,j))
            # print("id = " , idx_op)
            if self.constraints_config is not None:
                if self.constraints_config.get("job_with_no_skills", False):
                    worker_qualififed_for_op_curr = self.get_worker_at_most_1_level_for_operation((i, j))
                else :
                    worker_qualififed_for_op_curr = self.get_qualified_workers_for_operation((i, j))
            else:
                worker_qualififed_for_op_curr = self.get_qualified_workers_for_operation((i, j))


            # print("worker_qualififed_for_op_curr=", worker_qualififed_for_op_curr)
            if len(worker_qualififed_for_op_curr) > 0:
                worker_assignment[idx_op] = np.random.choice(worker_qualififed_for_op_curr)
            else :
                worker_assignment[idx_op] = -1 # we assign -1 if no worker is qualified for this operation
            
        return worker_assignment
         
    def assign_second_workers_to_operations(self, fwac, verbose=False):
        """ 
        Hypoithesis: the worker assigned to an operation must have at most a difference of 1 between its level and the level required by the operation (if worker do not have the competence) 
        Parameters:
        - sorted_operations: list of operations sorted in a way that respects the precedence constraints
        - worker_assignment: list of the first workers assigned to each operation

        Returns:
            - second_worker_assignment: list of the second workers assigned to each operation
        """
        swac = np.zeros(self.nb_op, dtype=int)
        for idx_op in range(len(swac)):
            
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            if verbose:
                print("operation=", (i,j))

            if fwac[idx_op] != -1 :

                task_idx_instance = self.instance.jobs_struct[i][j]

                                    
                # Si la tache ne peut etre fait en collaboration, ne pas ajouter un worker ayant le niveau
                if self.instance.tasks_times[task_idx_instance][2] == -1 :
                    worker_at_most_1_level_for_op_curr = self.get_worker_at_most_1_level_for_operation_and_not_qualified((i, j))
                else:
                    worker_at_most_1_level_for_op_curr = self.get_worker_at_most_1_level_for_operation((i, j))


                if len(worker_at_most_1_level_for_op_curr) > 0: 
                    w_selected = np.random.choice(worker_at_most_1_level_for_op_curr)
                    if w_selected != fwac[idx_op] :
                        swac[idx_op] = w_selected

                    else : 
                        swac[idx_op] = -1 
                else : 
                    swac[idx_op] = -1 
            else : 
                swac[idx_op] = -1

        return swac

    def get_qualified_workers_for_operation(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i
        Retourne la liste des workers qualifiés pour réaliser l'opération (i,j)
        Retourne liste vide si aucun worker n'est qualifié pour réaliser l'opération (i,j)
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where(self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id])[0]

    def get_worker_at_most_1_level_for_operation(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i
        Returns: workers list that have at most a difference of 1 between their level and the level required by the operation, otherwise empty list
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where(self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id] - LEVEL_DIFFERENCE)[0]

    def get_worker_at_most_1_level_for_operation_and_not_qualified(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i
        Utile pour les tasks qui ne peuvent etre fait en collab
        Returns: workers list that have at most a difference of 1 between their level and the level required by the operation and are not qualified for the operation, otherwise empty list
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where((self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id] - LEVEL_DIFFERENCE) & (self.instance.levels_workers[:,index_m] < self.instance.tasks_difficulties[task_id]))[0]

    def generate_random(self):

        osc = self.init_operations_order()
        fwac = self.init_first_worker_assignment(osc)
        swac = self.assign_second_workers_to_operations(fwac)

        return Chromosome(osc, fwac, swac)


class Chromosome:
    """ Représente une solution candidate """
    def __init__(
            self, 
            osc,
            fwac,
            swac,
            fitness = None,
            objectives = None
    ):
        self.osc = osc
        self.fwac = fwac
        self.swac = swac
        self.fitness = fitness
        self.objectives = objectives

    def copy(self):
        return Chromosome(
            self.osc.copy(),
            self.fwac.copy(),
            self.swac.copy(),
            self.fitness,
            self.objectives
        )

    def __str__(self):
        res = (f"osc :\n{self.osc}\n")
        res += (f"fwac:\n{self.fwac}\n")
        res += (f"swac:\n{self.swac}\n")
        if self.fitness is not None:
            res += (f"fitness: {self.fitness}\n")
        if self.objectives is not None:
            res += (f"objectives: {self.objectives}\n")
        return res


class Evaluator:
    """ Evalue la qualité d'un chromosome """
    def __init__(self, decoder, weights):
        self.decoder = decoder
        self.weights = weights

    def evaluate_agg(self, chromosome):
        obj1, obj2, obj3 = self.decoder.get_objectives(chromosome.osc, chromosome.fwac, chromosome.swac)
        fitness = (
            self.weights["profit"] * obj1 
            + self.weights["skills"] * obj2
            - self.weights["cognitive_load"] * obj3
        )
        chromosome.fitness = fitness
        chromosome.objectives = (obj1, obj2, obj3)
        return fitness




class Decoder:
    def __init__(self, instance, dicos, constraints_config=None):
        self.instance = instance
        self.dicos = dicos
        self.constraints_config = constraints_config or {}
        self.limit_makespan = self.constraints_config.get("constrained_makespan", -1)
        self.nb_op_in_job = np.array([len(self.instance.jobs_struct[i]) for i in range(self.instance.nb_jobs)])


    def _get_modes(self, osc, fwac, swac):
        """
        Retourne :
            - level_w1, [i] == 1 if w1 has level
                            == -1 if w1 do not have level
                            == 0 if task not assigned to any worker
                            numpy.ndarray of size nb_operations

            - level_w2, [i] == 1 if w2 has level
                            == -1 if w2 do not have level
                            == 0 if task not assigned to any worker or w2 == -1 (solo)
                            numpy.ndarray of size nb_operations
            
            - operation_mode, [i] == 0 alone
                                == 1 teaching
                                == 2 collaboration
                                == 3 alone no level
                                == -1 not assigned to any worker
                                numpy.ndarray of size nb_operations
        """
        level_w1 = np.zeros(len(osc), dtype=int)
        level_w2 = np.zeros(len(osc), dtype=int)
        operation_mode = np.zeros(len(osc), dtype=int) 

        
        for idx_op in range(len(osc)):
            
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            # print(f"O_{i}{j} : id=", idx_op)
            task_id_instance = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
            index_m = self.instance.task_to_m[task_id_instance] # index of the profession

            w1 = fwac[idx_op]
            w2 = swac[idx_op]

            l_task = self.instance.tasks_difficulties[task_id_instance] # level of the task
        
            # Si w1 dessus
            if w1 != -1: 
                
                l_w1 = self.instance.levels_workers[w1][index_m] # level of worker 1 for the profession of the task
                
                # si w1 pas niveau
                if l_w1 < l_task:
                    level_w1[idx_op] = -1
                    operation_mode[idx_op] = 3 # alone no level
                # si w1 a niveau
                else:
                    level_w1[idx_op] = 1

                
                # si w2 dessus  
                if w2 >= 0: # if w2 == -1 means that the task is done in solo by w1, if w2 == -2 means that the task is not assigned to any worker
                    l_w2 = self.instance.levels_workers[w2][index_m] 
                    
                    # si w2 pas niveau
                    if l_w2 < l_task: 
                        level_w2[idx_op] = -1
                        operation_mode[idx_op] = 1 # teaching
                    # si w2 a niveau
                    else:
                        level_w2[idx_op] = 1
                        operation_mode[idx_op] = 2 # collaboration

                # si w2 pas dessus
                else:
                    level_w2[idx_op] = 0
                    operation_mode[idx_op] = 0 # alone
            
            # si aucun worker dessus
            else :
                level_w1[idx_op] = 0
                level_w2[idx_op] = 0
                operation_mode[idx_op] = -1 # operation not assigned to any worker

        return level_w1, level_w2, operation_mode

    def _compute_skills(self, osc, fwac, swac, job_done_before_limit):
        """
        Calculate the skills evolution for each worker for each profession
        hypothesis: the assignment respect the rules of assignement (not 2 workers wihout level in a same task, etc...)
        """
        skills = self.instance.levels_workers.copy() # we initialize the skills with the initial levels of the workers

        for idx_op in range(len(osc)): # for each operation in the solution
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i

            if job_done_before_limit[i] == 1 : #on calcule skills sur les opérations des jobs se finissant avant limite makespan

                task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
                index_m = self.instance.task_to_m[task_id] # index of the profession

                w1 = fwac[idx_op]
                w2 = swac[idx_op]

                l_task = self.instance.tasks_difficulties[task_id] # level of the task
            
            

                if w1 != -1:
                    l_w1 = self.instance.levels_workers[w1][index_m] # level of worker 1 for the profession of the task
                    
                    
                    if l_w1 < l_task: # if worker 1 is not qualified for the task
                        skills[w1][index_m] += COEF_LEARNING 

                    
                    elif w2 >= 0: # if w2 == -1 means that the task is done in solo by w1, if w2 == -2 means that the task is not assigned to any worker
                        l_w2 = self.instance.levels_workers[w2][index_m] # level of worker 2 for the profession of the task

                        if l_w2 < l_task: # if w2 is not qualified for the task
                            skills[w2][index_m] = skills[w2][index_m] + COEF_LEARNING


        skills = np.minimum(self.instance.levels_workers + 1, skills) # skills can not exceed the initial level + 1 for one shift
        skills = np.minimum(LEVEL_MAX, skills) # skills can not exceed the maximum level
        return skills

    def _compute_cognitive_load(self, osc, fwac, swac,level_w1, level_w2, job_done_before_limit):

        """
        Calculate the cognitive load for each worker for each profession
        hypothesis: the assignment respect the rules of assignement (not 2 workers wihout level in a same task, etc...)
        """
        cognitive_load = np.zeros((self.instance.nb_workers, self.instance.nb_professions))

        for idx_op in range(len(osc)): # for each operation in the solution
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i

            if job_done_before_limit[i] == 1 : #on calcule mental load sur les opérations des jobs se finissant avant limite makespan

                task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
                index_m = self.instance.task_to_m[task_id] # index of the profession

                w1 = fwac[idx_op]
                w2 = swac[idx_op]

                # COLLAB
                if level_w1[idx_op] == 1 and level_w2[idx_op] == 1: # if both workers have the required level, they can collaborate
                    # mettre calcul du cognitive load pour la collaboration dans une fonction a part pour eviter de surcharger cette fonction
                    cognitive_load[w1][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_COLLAB * (LEVEL_MAX + 1 - self.instance.levels_workers[w1][index_m]) 
                    cognitive_load[w2][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_COLLAB * (LEVEL_MAX + 1 - self.instance.levels_workers[w2][index_m]) 
                
                # TEACHING
                elif level_w1[idx_op] == 1 and level_w2[idx_op] < 0: 
                    cognitive_load[w1][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_TUTOR * (LEVEL_MAX + 1 - self.instance.levels_workers[w1][index_m]) 
                    cognitive_load[w2][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_APPRENTI * (LEVEL_MAX + 1 - self.instance.levels_workers[w2][index_m])
                
                # ALONE NO LEVEL (pas de cognitive load pour le moment)
                # ALONE (pas de cognitive load pour le moment)

        return cognitive_load


    def _constrained_start_date_calculate(self, osc):
        """
        Retourne la liste des operation a faire avant celle à l'index i dans le chromosome pour respecter les contraintes de précédence
        """
        constained_operations = [ [] for _ in range(len(osc)) ]
        for idx_op in range(len(osc)):
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            prec = np.where(self.instance.constraints_precedence_operations[i,:,j] == 1)[0] # list of j of the precedence constraints for the operation (i,j)
            # print("\noperation=", (i,j), "prec=")
            for p in prec:
                # print(self.dicos.op_to_id[(i,p)], ":",i,p,end=" ")
                constained_operations[idx_op].append(self.dicos.op_to_id[(i,p)]) # we add the id of the operation (i,p) to the list of constrained operations for the operation (i,j)

        return constained_operations
    
    def _count_nb_tasks_done_by_worker(self, fwac, swac):
        """
        Count the number of tasks done by each worker in the solution
        """
        assignment = np.concatenate((fwac, swac))
        worker, count = np.unique(assignment, return_counts=True)

        idx_to_remove = np.where(worker < 0) # on enlève les workers -1 (solo) et -2 (not assigned) du comptage
        worker = np.delete(worker, idx_to_remove)
        count = np.delete(count, idx_to_remove)

        return worker, count

    def _start_date_calculate(self, osc, fwac, swac):
        """
        Calculate the start date of each operation in the solution (maximum left possible)
        """
        # Nombre maximal de tache de tach fait par 1 worker
        nb_task_by_w = self._count_nb_tasks_done_by_worker(fwac, swac)
        
        constrained_op = self._constrained_start_date_calculate(osc)
        # print("constrained op =", constrained_op)
        _, _, mode = self._get_modes(osc, fwac, swac)
        
        start_worker_id = np.zeros((self.instance.nb_workers, len(osc)), dtype=float) # [w][i] = date time of task i of worker w, otherwise 0
        start_worker_id.fill(-1)
        #      0.  1.  2.  3.  4.  
        # w0   7   0   3  -1  -1  
        # w1  -1  -1   3  -1  -1  
        # w2  -1  -1  -1  -1   0

        finish_worker_id = np.zeros((self.instance.nb_workers, len(osc)), dtype=float) # [w][i] = finish time of task i of worker w, otherwise 0
        finish_worker_id.fill(-1)
        

        number_viewed_job = np.zeros(self.instance.nb_jobs, dtype=int) # pour savoir quelle op du job à étée deja vue
        ECART = 0 # 0.01
        for pos_op in range(len(osc)):
            
            i = osc[pos_op] # job i
            j = number_viewed_job[i] # operation j of job i
            number_viewed_job[i] += 1
            idx_op = self.dicos.op_to_id[(i,j)] # index of the operation (i,j) in the chromosome

            # (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            instance_task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
            # print("idx_op=", idx_op, "operation=", (i,j), "instance_task_id=", instance_task_id)
            # print(fwac[idx_op], swac[idx_op])



            w1 = fwac[idx_op]
            w2 = swac[idx_op]

            # print("--------")
            # print(f"idx_op= {idx_op}, 0({i},{j}), w1={w1}, w2={w2}, mode={mode[idx_op]}")


            ####### ALONE #######
            if mode[idx_op] == 0:
                # print("ALONE")
                ############### INDEX TASKS MUST DONE BEFORE ##############
                finish_time_before = 0
                all_id_before = constrained_op[idx_op] 
                # print("all_id_before=", all_id_before)
                if len(all_id_before) > 0:
                    finish_time_before = np.max(finish_worker_id[:, all_id_before], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                    # !!! Si cette tache qui doit etre fait avant n'a pas été affecté alors ne pas affecter cette tache courante !!!
                    if finish_time_before == -1 : 
                        start_worker_id[w1][idx_op] = -1
                        finish_worker_id[w1][idx_op] = -1
                        continue


                    # print("finish_time_before=", finish_time_before)

                finish_by_w = np.max(finish_worker_id[w1]) # recupérer la date de fin de la dernière tâche faite par w1
                # print("finish_by_w=", finish_by_w)
                start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before) + ECART # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
                finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][0] # calculer la date de fin de la tâche actuelle pour w1
                # print("start_worker_id=", start_worker_id[w1][idx_op], "finish_worker_id=", finish_worker_id[w1][idx_op])

            ####### TEACHING #######
            elif mode[idx_op] == 1:
                # print("TEACHING")
                finish_time_before = 0
                all_id_before = constrained_op[idx_op]
                # print("all_id_before=", all_id_before)
                if len(all_id_before) > 0:

                    finish_time_before = np.max(finish_worker_id[:, all_id_before], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                    if finish_time_before == -1 : 
                        start_worker_id[w1][idx_op] = -1
                        finish_worker_id[w1][idx_op] = -1
                        continue
                    # print("finish_time_before=", finish_time_before)

                finish_by_w = max(np.max(finish_worker_id[w1]), np.max(finish_worker_id[w2])) # date de fin la moins tardive pour commencer task curr
                # print("finish_by_w=", finish_by_w)
                start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before) + ECART # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
                start_worker_id[w2][idx_op] = start_worker_id[w1][idx_op] # commencer en même temps que w1
                finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][1]
                finish_worker_id[w2][idx_op] = start_worker_id[w2][idx_op] + self.instance.tasks_times[instance_task_id][1]
                # print("start_worker_id=", start_worker_id[w1][idx_op], "finish_worker_id=", finish_worker_id[w1][idx_op])

            ####### COLLABORATION #######
            elif mode[idx_op] == 2: # collaboration
                # print("COLLABORATION")
                finish_time_before = 0
                all_id_before = constrained_op[idx_op]
                # print("all_id_before=", all_id_before)
                if len(all_id_before) > 0:
                    finish_time_before = np.max(finish_worker_id[:, all_id_before], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                    if finish_time_before == -1 : 
                        start_worker_id[w1][idx_op] = -1
                        finish_worker_id[w1][idx_op] = -1
                        continue
                    # print("finish_time_before=", finish_time_before)

                finish_by_w = max(np.max(finish_worker_id[w1]), np.max(finish_worker_id[w2])) # date de fin la moins tardive pour commencer task curr
                # print("finish_by_w=", finish_by_w)
                start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before) + ECART # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
                start_worker_id[w2][idx_op] = start_worker_id[w1][idx_op]
                
                finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][2]
                finish_worker_id[w2][idx_op] = finish_worker_id[w1][idx_op]
                # print("start_worker_id=", start_worker_id[w1][idx_op], "finish_worker_id=", finish_worker_id[w1][idx_op])

            ####### ALONE NO LEVEL #######
            elif mode[idx_op] == 3: # alone no level
                # print("ALONE NO LEVEL")
                finish_time_before = 0
                all_id_before = constrained_op[idx_op]
                # print("all_id_before=", all_id_before)
                if len(all_id_before) > 0:
                    finish_time_before = np.max(finish_worker_id[:, all_id_before], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                    if finish_time_before == -1 : 
                        start_worker_id[w1][idx_op] = -1
                        finish_worker_id[w1][idx_op] = -1
                        continue
                    # print("finish_time_before=", finish_time_before)

                finish_by_w = np.max(finish_worker_id[w1]) # recupérer la date de fin de la dernière tâche faite par w1
                # print("finish_by_w=", finish_by_w)
                start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before) + ECART # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
                finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][2] * PERC_SOLO_NO_LEVEL_TIME
                # print("start_worker_id=", start_worker_id[w1][idx_op], "finish_worker_id=", finish_worker_id[w1][idx_op])

        return start_worker_id, finish_worker_id

    def _compute_job_end_dates(self, finish_worker_id):
        # print("finish_worker_id=", finish_worker_id)

        end_date_job = np.zeros(self.instance.nb_jobs)
        for i in range(self.instance.nb_jobs):
            nb_op_job_i = len(self.instance.jobs_struct[i])
            idx_ops_job = np.zeros(nb_op_job_i, dtype=int) # idx des ops du job i

            for j in range(nb_op_job_i):
                idx_ops_job[j] = self.dicos.op_to_id[(i,j)]

            end_date_job[i] = np.max(finish_worker_id[:, idx_ops_job]) # dat de fin du job i

        return end_date_job

    def _compute_job_done(self, fwac):
        
        job_done = np.ones(self.instance.nb_jobs, dtype=int)
        
        for i in range(len(job_done)):
            for j in range(len(self.instance.jobs_struct[i])):
                idx_op = self.dicos.op_to_id[(i,j)]
                if fwac[idx_op] == -1 :
                    job_done[i] = 0
                    break

        return job_done

    def _compute_job_done_before_limit(self, job_end_dates):
        job_done_before_limit = np.ones(self.instance.nb_jobs, dtype=int)
        
        for i in range(len(job_done_before_limit)):
            if job_end_dates[i] > self.limit_makespan:
                job_done_before_limit[i] = 0

        return job_done_before_limit

    def _compute_task_not_possible_done(self, osc, fwac, swac):
        """
        Dans le vecteur osc, les tasks affecté telle que la tache qui doit etre fait avant celle la n'est pas affecté, 
        ne doivent pas être affecté
        """
        for i in range(self.instance.nb_jobs):

            idx_job_i = np.zeros(self.nb_op_in_job[i], dtype=int) # idx des ops du job i
            for j in range(self.nb_op_in_job[i]):
                    idx_job_i[j] = self.dicos.op_to_id[(i,j)]

            if np.any(fwac[idx_job_i] == -1) : # Si une des opérations du job i n'est pas affecté, alors toutes les op de ce job pas affecté
                fwac[idx_job_i] = -1
                swac[idx_job_i] = -1 # pas necessaire normalement


    def decode_fast(self, osc, fwac, swac):
        """
        Sert pour calculer rapidement la fitness d'un chromosome sans calculer tous les éléments de la solution
        """

        
        
        self._compute_task_not_possible_done(osc, fwac, swac)
        lw1, lw2, modes = self._get_modes(osc, fwac, swac)
        start, finish = self._start_date_calculate(osc, fwac, swac)

        job_done = self._compute_job_done(fwac)
        job_end_dates = self._compute_job_end_dates(finish)
        job_done_before_limit = self._compute_job_done_before_limit(job_end_dates)


        skills = self._compute_skills(osc, fwac, swac, job_done_before_limit)

        cognitive_load = self._compute_cognitive_load(osc, fwac, swac, lw1, lw2, job_done_before_limit)

        return {
            "start": start,
            "finish": finish,
            "skills": skills,
            "cognitive_load": cognitive_load,
            "lw1": lw1,
            "lw2": lw2,
            "modes": modes,
            "job_done": job_done,
            "job_end_dates": job_end_dates,
            "job_done_before_limit": job_done_before_limit
        }

    def get_objectives(self, osc, fwac, swac):
        decoded = self.decode_fast(osc, fwac, swac)

        obj1 = np.sum(decoded["job_done_before_limit"] * self.instance.resale_price_jobs)# * decoded["job_done"])
        obj2 = np.sum(decoded["skills"]) - np.sum(self.instance.levels_workers)
        obj3 = np.sum(decoded["cognitive_load"])
        return [obj1, obj2, obj3]

    def decode_solution(self, osc, fwac, swac):
        """
        Sert pour calculer rapidement la fitness d'un chromosome sans calculer tous les éléments de la solution
        """        
        decoded = self.decode_fast(osc, fwac, swac)


        makespan = np.max(decoded["finish"])
        # print("makespan=", makespan)
        # print("limit_makespan=", limit_makespan)
        # print("job_done=", job_done)
        # print("finish_time_job=", finish_time_job)
        # print("job_done_before_limit=", job_done_before_limit)

        
        s = Solution(self.instance)
        for idx_op in range(len(osc)):
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            w1 = fwac[idx_op]
            w2 = swac[idx_op]
            mode_op = decoded["modes"][idx_op]

            if w1 != -1:
                s.x[i,j,w1] = 1
                s.d[i,j,w1] = decoded["start"][w1][idx_op]
                s.f[i,j,w1] = decoded["finish"][w1][idx_op]
                s.z_auxilary[i,j,mode_op] = 1
                
                if mode_op == 1: # teaching (tutor)
                    s.is_tutor[i,j, w1] = 1

            if w2 != -1:
                s.x[i,j,w2] = 1
                s.d[i,j,w2] = decoded["start"][w2][idx_op]
                s.f[i,j,w2] = decoded["finish"][w2][idx_op]
                s.z_auxilary[i,j,mode_op] = 1

            
            
        s.l = decoded["skills"]
        s.cognitive_load_total = decoded["cognitive_load"]
        s.job_done = decoded["job_done"]
        s.job_done_before_limit = decoded["job_done_before_limit"]
        s.borne_sup_makespan = self.limit_makespan

        return s




class Neighborhood:
    def generate(self, chromosome):
        pass

# stochastique
class SwapOperationNeighborhood(Neighborhood):
    def __init__(self, n):
        self.n = n

    def generate(self, chromosome):
        """ Génere n voisins en permutant 2 opérations """
        voisins = []
        o1 = np.random.choice(len(chromosome.osc), size=self.n, replace=True)
        o2 = np.random.choice(len(chromosome.osc), size=self.n, replace=True)
        for idx_1, idx_2 in zip(o1, o2):
            v = chromosome.copy()
            v.osc[idx_1], v.osc[idx_2] = v.osc[idx_2], v.osc[idx_1]
            voisins.append(v)
        return voisins

# stochastique dans le choix de l'opération à déplacer, mais déterministe dans le choix de la position où la déplacer (toutes les positions possibles sont testées)
class MoveOneOperationNeighborhood(Neighborhood): 
    def __init__(self):
        pass

    def generate(self, chromosome) :
        """ Choisi une operation et la déplace à une autre position dans l'ordre de séquencement """
        voisins = []
        idx = np.random.choice(len(chromosome.osc), replace=True)
        # print("job elu = ", chromosome.osc[idx])
        # print("idx =", idx)

        for iter in range(len(chromosome.osc)):
            if chromosome.osc[iter] != chromosome.osc[idx]: # on ne fait pas de permutation si c'est le même job
                v = chromosome.copy()
                tmp = v.osc[iter]
                v.osc[iter] = v.osc[idx]
                v.osc[idx] = tmp
                
                voisins.append(v)
        return voisins

# stochastique
class SwapFixedOperationsNeighborhood(Neighborhood):
    def __init__(self, k, n):
        self.k = k # nombre d'opérations fixées
        self.n = n # nombre de voisins à générer

    def generate(self, chromosome) :
        """ k operation fixées et les autres permutées n fois pour générer n voisins """
        voisins = []
        idx_fixed = np.random.choice(len(chromosome.osc), size=self.k, replace=True)
        idx_to_permute = [i for i in range(len(chromosome.osc)) if i not in idx_fixed]
        for _ in range(self.n):
            v = chromosome.copy()
            tmp = v.osc[idx_to_permute].copy()
            np.random.shuffle(tmp)
            v.osc[idx_to_permute] = tmp
            voisins.append(v)
        return voisins
        
# stochastique
class ShuffleOperationsNeighborhood(Neighborhood):
    def __init__(self, n):
        self.n = n # nombre de voisins à générer

    def generate(self, chromosome) :
        """ Génere n voisins en faisant des modifications aléatoires sur le chromosome """
        voisins = []
        for _ in range(self.n):
            v = chromosome.copy()
            v.osc = np.random.permutation(v.osc)
            voisins.append(v)
        return voisins

# deterministe
class MoveSecondWorkerNeighborhood(Neighborhood):
    def __init__(self, generator):
        self.generator = generator

    def _get_possible_workers_for_operation(self, idx_op):
        """ Retourne la liste des workers possibles pour être le second worker de l'opération idx_op du chromosome """
        (i,j) = self.generator.dicos.id_to_op[idx_op] # operation j of job i
        idx_task_instance = self.generator.instance.jobs_struct[i][j] # task id for know information about the task in the instance

        if self.generator.instance.tasks_times[idx_task_instance][2] == -1 : # Pas de collaboration possible pour cette tâche
            idx_workers = self.generator.get_worker_at_most_1_level_for_operation_and_not_qualified((i,j))
        else:
            idx_workers = self.generator.get_worker_at_most_1_level_for_operation_and_not_qualified((i,j))

        idx_workers = np.append(idx_workers, -1) # on ajoute la possibilité de ne pas mettre de second worker (solo)
        return idx_workers

    def generate(self, chromosome) :
        """ Choisi une operation et change le worker 2 de cette opération parmis tout ceux possibles """
        voisins = []
        idx = np.random.choice(len(chromosome.osc), replace=True)
        workers_possible = self._get_possible_workers_for_operation(idx)
        if len(workers_possible) == 0:
            return voisins
        for w in workers_possible:
            if w != chromosome.swac[idx]: # on ne fait pas de permutation si c'est le même worker
                v = chromosome.copy()
                v.swac[idx] = w
                voisins.append(v)
        return voisins

# Fonction de voisinage à tester
# - alterner entre swap deux op et change worker swac


class Selector:
    def select(self, fitness):
        pass

class SoftmaxSelector(Selector):
    def __init__(self):
        pass

    def select(self, fitness):
        probs = np.exp(fitness - np.max(fitness)) / np.sum(np.exp(fitness - np.max(fitness)))
        return np.random.choice(len(fitness), p=probs)

class BestSelector(Selector):
    def __init__(self):
        pass

    def select(self, fitness):
        return np.argmax(fitness)


class RandomSearch:
    """ Algorithme de recherche aléatoire """
    def __init__(self, generator, evaluator):
        self.generator: ChromosomeGenerator = generator
        self.evaluator: Evaluator = evaluator

    def run(self, max_iter, verbose=False):

        best_list = []
        all_viewed_list = []
        i_find_best_list = [] # itération où l'on a trouvé le meilleur chromosome jusqu'à présent 
        best_fitness = -float('inf')

        for i in range(max_iter):
            chromosome : Chromosome = self.generator.generate_random()
            fitness : float = self.evaluator.evaluate_agg(chromosome)
            chromosome.fitness = fitness
            all_viewed_list.append(chromosome)
            
            if fitness > best_fitness:
                if verbose:
                    print(f"iter {i}: New best fitness: {fitness}")
                best_fitness = fitness
                best_list.append(chromosome)
                i_find_best_list.append(i)

        return all_viewed_list, best_list, i_find_best_list



class LocalSearch:
    def __init__(self, generator, evaluator, neighborhood, selector):
        self.generator: ChromosomeGenerator = generator
        self.evaluator: Evaluator = evaluator
        self.neighborhood: Neighborhood = neighborhood
        self.selector: Selector = selector

    def run(self, max_iter, verbose=False):
        """
        Génere un chromosome random, puis génère des voisins et séléctionne "le meilleur" voisin puis itère le processus
        """
        
        chromosome = self.generator.generate_random()
        chromosome.fitness = self.evaluator.evaluate_agg(chromosome)
        if verbose:
            print(f"Initial fitness: {chromosome.fitness}")
            # print(chromosome)
        
        best_list = [chromosome]
        all_viewed_list = [chromosome]

        
        for i in range(1, max_iter):
            # print(f"iter :{i} - c.fitness = {chromosome.fitness}")
            voisins = self.neighborhood.generate(chromosome)
            if len(voisins) == 0: # le cas si la fonction de voisinage est deterministe
                # if verbose:
                #     print(f"iter {i}: No neighbors generated, stopping search.")
                voisins = [chromosome] # on reste sur le même chromosome
            fitness_v = np.array([self.evaluator.evaluate_agg(v) for v in voisins])

            selected_idx = self.selector.select(fitness_v)

            if voisins[selected_idx].fitness >= chromosome.fitness:
                chromosome = voisins[selected_idx]
                chromosome.fitness = fitness_v[selected_idx]

            # chromosome = voisins[selected_idx]
            # chromosome.fitness = fitness_v[selected_idx]
            
            # best si superieur strict seulement
            if chromosome.fitness > best_list[-1].fitness:
                if verbose:
                    print(f"iter {i}: New best fitness: {chromosome.fitness}")
            
                best_list.append(chromosome)
            all_viewed_list.append(chromosome)
        return all_viewed_list, best_list