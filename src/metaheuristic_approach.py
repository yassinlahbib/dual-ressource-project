from Instance import *
from Solution import *
from Constante import *
from copy import copy, deepcopy
import tqdm


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


def assert_base_worker_is_respcted(x, base_worker_assignment, nb_job_no_skills, jobs_struct):
    assert np.sum(base_worker_assignment[1]) <= nb_job_no_skills
    for i in range(len(jobs_struct)):
        if base_worker_assignment[0, i] >= 0:
            w = base_worker_assignment[0, i]
            for j in range(len(jobs_struct[i])):
                workers_done_op = np.where(x[i, j, :] >= 1)[0]
                if len(workers_done_op) > 0 :
                    assert w in workers_done_op, f"Base worker {w} not respected for job {i}, operation {j}, bwa={base_worker_assignment}, nb_op_in_job={len(jobs_struct[i])}, x={x[i, j, :]}"
    return True


def chromosome_to_x_variable(chromosome, instance, dicos):
    """ Convert a chromosome to a vector of x variables like the MILP model """

    # print(chromosome)
    x = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers), dtype=int)

    for idx_op in range(len(chromosome.fwac)):
        # print("idx_op=", idx_op)
        (i,j) = dicos.id_to_op[idx_op] # operation j of job i
        if chromosome.fwac[idx_op] >= 0:
            w1 = chromosome.fwac[idx_op]
            # print("w1=", w1)
            x[i,j,w1] = 1
        if chromosome.swac[idx_op] >= 0:
            w2 = chromosome.swac[idx_op]
            # print("w2=", w2)
            x[i,j,w2] = 1
    return x

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
        Retourne un vecteur de taille nb_operations, qui contient l'id des jobs dans un ordre topologique (preserve les contraintes de précédence)
        """
        operations_order = []        
        for i in range(len(self.nb_op_in_job)):
            operations_order.extend([i] * self.nb_op_in_job[i])
        
        operations_order = np.array(operations_order)
        np.random.shuffle(operations_order)

        return operations_order
    
    def init_job_order(self):
        """
        Retourne un vecteur de taille nb_operations, qui contient l'id des jobs
        Un job est consécutif pour toutes ses opérations, mais l'ordre des jobs est aléatoire
        """
        job_order_shuffled = np.random.permutation(self.instance.nb_jobs)
        operations_order = []
        for i in range(len(job_order_shuffled)):
            job_id = job_order_shuffled[i]
            operations_order.extend([job_id] * self.nb_op_in_job[job_id])

        return np.array(operations_order)

    def init_job_order_by_resale_price(self):
        """
        Retourne un vecteur de taille nb_operations, qui contient l'id des jobs
        ordonnés par prix de revente décroissant des jobs et les opérations d'un job sont consécutives
        """
        job_order = np.argsort(self.instance.resale_price_jobs)[::-1]
        operations_order = []
        for job_id in job_order:
            operations_order.extend([job_id] * self.nb_op_in_job[job_id])

        return np.array(operations_order)
        
   
         

    def get_qualified_workers_for_operation(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i 
        Retourne la liste des workers qualifiés pour réaliser l'opération (i,j) 
        et liste vide si aucun worker n'est qualifié pour réaliser l'opération (i,j)
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where(self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id])[0]

    def get_worker_at_most_1_level_for_operation(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i
        Returns: workers list that have at most a difference of 1 between their level and the level required by the operation (qualified or not qualified), otherwise empty list
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where(self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id] - LEVEL_DIFFERENCE)[0]

    def get_worker_at_most_1_level_for_operation_and_not_qualified(self, operation_struct):
        """
        operation_struct: (i,j) operation j of job i
        Returns: workers list that have at most a difference of 1 between their level and the level required by the operation AND are not qualified for the operation, otherwise empty list
        """
        (i, j) = operation_struct
        task_id = self.instance.jobs_struct[i][j]
        index_m = self.instance.task_to_m[task_id]
        return np.where((self.instance.levels_workers[:,index_m] >= self.instance.tasks_difficulties[task_id] - LEVEL_DIFFERENCE) & (self.instance.levels_workers[:,index_m] < self.instance.tasks_difficulties[task_id]))[0]


    def calculate_matrix_difference_level_tasks_workers(self, verbose=False):
        """ Calcule la matrice de difference de niveau entre les taches et les workers pour savoir le niveau de difference entre toutes les opérations et tous les workers 
        
            Returns: np.ndarray (nb_jobs, nb_operations, nb_workers)
                matrice de difference de niveau entre les taches et les workers
        """
        difference_level_tasks_workers = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.nb_workers))
        for i in range(len(self.instance.jobs_struct)):
            
            for j in range(len(self.instance.jobs_struct[i])):
                task_id = self.instance.jobs_struct[i][j]
                index_m = self.instance.task_to_m[task_id]
                for k in range(self.instance.nb_workers):
                    # print("worker=", k, "task=", task_id, "level_worker=", self.instance.levels_workers[k, index_m], "level_task=", self.instance.tasks_difficulties[task_id])
                    # print("diff=", self.instance.levels_workers[k, index_m] - self.instance.tasks_difficulties[task_id])
                    # print("\n")
                    difference_level_tasks_workers[i, j, k] = self.instance.levels_workers[k, index_m] - self.instance.tasks_difficulties[task_id]

        if verbose:
            print("matrice difference level tasks workers:")
            print(difference_level_tasks_workers)
        return difference_level_tasks_workers





    #IF WORKER DE BASE IS ACTIVATED
    def assign_fwac_swac_with_base_worker(self, base_worker_assignment, operations_order):
        """ On peut faire les operations d'un job meme s'il n'est pas done et il faut quand meme un worker de base 
        mais possible qu'il ne puisse pas faire toutes les operation du job 
        dans ce cas la le job ne sera pas done mais aura quand meme un worker de base 
        ON ne prend pas en compte worker solo no level pour le moment """

        fwac = np.zeros(self.nb_op, dtype=int) #toujours le worker savec le plus grand niveau  pour l'opérations
        swac = np.zeros(self.nb_op, dtype=int)
        cannot_done = False # pour savoir si l'opération ne peut pas etre faite car une operation precedente n'est pas faite

        for idx_op in range(len(operations_order)):


            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            id_task_instance = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
            metier = self.instance.task_to_m[id_task_instance] # index of the profession

            op_prec = np.where(self.instance.constraints_precedence_operations[i, :, j] == 1)[0] # operations that must be done before operation (i,j)
            if len(op_prec) > 0:
                for op in op_prec:
                    idx_op_prec = self.dicos.op_to_id[(i, op)]
                    if fwac[idx_op_prec] == -1: # if the precedent operation is not done, then the current operation cannot be done
                        # print("operation (", i, ",", j, ") NO (", i, ",", op, ") is not done")
                        fwac[idx_op] = -1
                        swac[idx_op] = -1
                        cannot_done = True
                        break
            # print('i=', i, 'j=', j, "len(op_prec)=", len(op_prec), "cannot_done=", cannot_done, "fwac[idx_op]=", fwac[idx_op], "swac[idx_op]=", swac[idx_op])
            # if cannot_done == False:
            #     print("operation (", i, ",", j, ") YES")
                
            if cannot_done:
                cannot_done = False
                continue # if the precedent operation is not done, then the current operation cannot be done

            if base_worker_assignment[0, i] >= 0: # if the job has a base worker assigned
                
                
                if base_worker_assignment[1, i] == 0: # if the job has a base worker assigned and it is a worker with level required
                    #choisir entre le mettre seul ou avec un autre worker (solo, collab, teaching)
                    
                    w_i = base_worker_assignment[0, i]
                    
                    ## Si ne peut pas etre fait en collab: pas de worker dans swac
                    # if REPRENDRE ICCCCII
                    if self.instance.tasks_times[id_task_instance][2] == -1 : # Si la tache ne peut etre fait en collab 
                        if self.instance.levels_workers[w_i][metier] >= self.instance.tasks_difficulties[id_task_instance]:# Si le worker de base a le niveau requis pour l'opération
                            fwac[idx_op] = w_i
                        else:
                            fwac[idx_op] = -1
                        swac[idx_op] = -1
                        continue

                    else :
                        if self.instance.levels_workers[w_i][metier] < self.instance.tasks_difficulties[id_task_instance] : # Si worker de base n'a pas le niveau requis pour l'opération
                            if self.instance.levels_workers[w_i][metier] + LEVEL_DIFFERENCE >= self.instance.tasks_difficulties[id_task_instance]: #On regarde le gap de difference 
                                # gap respecté, on peut chercher un worker qualifié pour l'opération
                                w_j = self.get_qualified_workers_for_operation((i, j))
                            else: #gap pas respecté, pas d'affectation pour l'opération
                                fwac[idx_op] = -1
                                swac[idx_op] = -1
                                continue
                        else :
                            w_j = self.get_worker_at_most_1_level_for_operation((i, j))

                        w_j = np.random.choice(w_j) # (len > 0 car on a un worker de base pour le job i -> plus vrai mtn) -> ajout de continue plus haut donc c bon

                        if self.instance.levels_workers[w_i][metier] >= self.instance.tasks_difficulties[id_task_instance] : # Si le worker de base a le niveau requis pour l'opération
                            if np.random.rand() < 0.5 : # 50% de chance de mettre le worker de base dans fwac ou swac
                                w_j = w_i # il fait en solo a 50 % de chance

                        if w_j == w_i :
                            fwac[idx_op] = w_i
                            swac[idx_op] = -1
                        else :
                            if self.instance.levels_workers[w_i][metier] >= self.instance.levels_workers[w_j][metier] :
                                fwac[idx_op] = w_i # fwac celui qui a le plus grand niveau pour l'opération
                                swac[idx_op] = w_j
                            else :
                                fwac[idx_op] = w_j
                                swac[idx_op] = w_i
            else: #if not base worker
                fwac[idx_op] = -1
                swac[idx_op] = -1



        return fwac, swac

    def generate_random(self, nb_job_no_skills=0):
        # print("In generate_random ---> nb_job_no_skills=", nb_job_no_skills)

        # self.osc = self.init_operations_order()
        self.osc = self.init_job_order()
        # self.osc = self.init_job_order_by_resale_price()



        difference_level_tasks_workers = self.calculate_matrix_difference_level_tasks_workers()


        self.base_worker_assignment = [[-1 for _ in range(self.instance.nb_jobs)],[ 0 for _ in range(self.instance.nb_jobs)]] # on initialise la liste des workers de base à -1 pour chaque job, et le nombre de jobs sans skills à 0
        self.base_worker_assignment = np.array(self.base_worker_assignment)
        # print("base_worker_assignment before =")
        # print(self.base_worker_assignment)
        #== AFFECTATION WORKER BASE ####PLUS BESOIN DE CLASSIFY ETC CAR NO AVAILABLE SOLO NO LEVEL
        #Recuperer les workers ayant un level gap d'au moins 1 avec la premiere op de chaque job

        level_gap = self.calculate_matrix_difference_level_tasks_workers()
        worker_index_available_base_worker = level_gap[:,0] >=-1
        worker_index_available_base_worker = [np.where(worker_index_available_base_worker[i])[0] for i in range(self.instance.nb_jobs)]

        for i in range(self.instance.nb_jobs):
            if len(worker_index_available_base_worker[i]) > 0:
                w = np.random.choice(worker_index_available_base_worker[i])
                self.base_worker_assignment[0][i] = w

        #== AFFECTATION FWAC ET SWAC
        fwac, swac = self.assign_fwac_swac_with_base_worker(self.base_worker_assignment, self.osc)

        self.fwac = fwac
        self.swac = swac

        #== MISE EN FORME DE LA VARIABLE D'AFFECTAION X
        chromosome = Chromosome(self.osc, self.fwac, self.swac)
        x = chromosome_to_x_variable(chromosome, self.instance, self.dicos)        
        self.x = x
        
        #== TEST DE RESPECT DU WORKER DE BASE
        # print("chromosome : ", chromosome)

        # print("x=")
        # print(x)
        res = assert_base_worker_is_respcted(x, self.base_worker_assignment, nb_job_no_skills=nb_job_no_skills, jobs_struct=self.instance.jobs_struct)
        # print("worker base is respected ?: ", res)
        chromosome.base_worker_assignment = self.base_worker_assignment
        return chromosome




class Chromosome:
    """ Représente une solution candidate """
    def __init__(
            self,
            osc,
            fwac,
            swac,
            fitness = None,
            objectives = None,
            base_worker_assignment = None,
            job_done = None,
    ):
        self.osc = osc
        self.fwac = fwac
        self.swac = swac
        self.fitness = fitness
        self.objectives = objectives
        self.base_worker_assignment = base_worker_assignment
        self.job_done = job_done

    def copy(self):
        return Chromosome(
            self.osc.copy(),
            self.fwac.copy(),
            self.swac.copy(),
            self.fitness,
            self.objectives,
            self.base_worker_assignment.copy() if self.base_worker_assignment is not None else None,
            self.job_done.copy() if self.job_done is not None else None,
        )

    def __str__(self):
        res = (f"osc :\n{self.osc}\n")
        res += (f"fwac:\n{self.fwac}\n")
        res += (f"swac:\n{self.swac}\n")
        if self.fitness is not None:
            res += (f"fitness: {self.fitness}\n")
        if self.objectives is not None:
            res += (f"objectives: {self.objectives}\n")
        if self.base_worker_assignment is not None:
            res += (f"base_worker_assignment:\n{self.base_worker_assignment}\n")
        if self.job_done is not None:
            res += (f"job_done:\n{self.job_done}\n")
        return res


class Evaluator:
    """ Evalue la qualité d'un chromosome """
    def __init__(self, decoder, weights):
        self.decoder = decoder
        self.weights = weights
        self.nb_evaluations = 0

    def get_nb_evaluations(self):
        return self.nb_evaluations

    def reset_evaluations(self):
        self.nb_evaluations = 0

    def evaluate_agg(self, chromosome, ponderation_task_done="one"):
        # print("in AGG :")
        # print(chromosome)
        obj1, obj2, obj3 = self.decoder.get_objectives(chromosome, ponderation_task_done=ponderation_task_done)
        # print(f"Objectives: profit={obj1}, skills={obj2}, cognitive_load={obj3} \n\n")
        fitness = (
            self.weights["profit"] * obj1 
            + self.weights["skills"] * obj2
            - self.weights["cognitive_load"] * obj3
        )

        chromosome.fitness = fitness
        chromosome.objectives = (obj1, obj2, -obj3)
        chromosome.job_done = self.decoder.decoded["job_done"].copy()
        self.nb_evaluations += 1
        return fitness


class Decoder:
    def __init__(self, instance, dicos, constraints_config=None):
        self.instance = instance
        self.dicos = dicos
        self.constraints_config = constraints_config or {}
        self.limit_makespan = self.constraints_config.get("constrained_makespan", -1)
        self.nb_op_in_job = np.array([len(self.instance.jobs_struct[i]) for i in range(self.instance.nb_jobs)])


# donne le mode opératoir de toute la sequence
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
                    operation_mode[idx_op] = 0 # alone pour l'instant

                
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

                ## NO NEED THIS PART NOW
                # # si w2 pas dessus
                # else:
                #     level_w2[idx_op] = 0
                #     operation_mode[idx_op] = 0 # alone
            
            # si aucun worker dessus
            else :
                level_w1[idx_op] = 0
                level_w2[idx_op] = 0
                operation_mode[idx_op] = -1 # operation not assigned to any worker

        return level_w1, level_w2, operation_mode

# donne le mode opératoire d'une opération
    def _get_mode_operation(self, task_id_instance, w1, w2):
        """
        Retourne le mode opératoire d'une opération
                == 0 alone
                == 1 teaching
                == 2 collaboration
                == 3 alone no level
                == -1 not assigned to any worker
        """
        index_m = self.instance.task_to_m[task_id_instance] # index of the profession
        l_task = self.instance.tasks_difficulties[task_id_instance] # level of the task
    
        # Si w1 dessus
        if w1 != -1: 
            
            l_w1 = self.instance.levels_workers[w1][index_m] # level of worker 1 for the profession of the task
            
            # si w1 pas niveau : w1 est le worker le plus qualifié pour l'opération, donc si il n'a pas le niveau, alors l'opération est faite seul sans niveau
            if l_w1 < l_task:
                assert w2 == -1, "If w1 does not have the level then no second worker should be assigned to the operation"
                return 3 # alone no level
            # si w1 a niveau
            else:
                # si w2 dessus  
                if w2 >= 0: 
                    l_w2 = self.instance.levels_workers[w2][index_m] 
                    
                    # si w2 pas niveau
                    if l_w2 < l_task: 
                        return 1 # teaching
                    # si w2 a niveau
                    else:
                        return 2 # collaboration

                # si w2 pas dessus
                else:
                    return 0 # alone
        
        # si aucun worker dessus
        else :
            return -1 # operation not assigned to any worker


    def _compute_skills(self, osc, fwac, swac, task_done):
        """
        Calculate the skills evolution for each worker for each profession
        hypothesis: the assignment respect the rules of assignement (not 2 workers wihout level in a same task, etc...)
        """
        skills = self.instance.levels_workers.copy() # we initialize the skills with the initial levels of the workers

        for idx_op in range(len(osc)): # for each operation in the solution
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i

            if task_done[i][j] == 1 : #on calcule skills sur les opérations faites

                task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
                index_m = self.instance.task_to_m[task_id] # index of the profession

                w1 = fwac[idx_op]
                w2 = swac[idx_op]

                l_task = self.instance.tasks_difficulties[task_id] # level of the task
            
            

                if w1 != -1:
                    lev_w1 = self.instance.levels_workers[w1][index_m] # level of worker 1 for the profession of the task
                    
                    
                    if lev_w1 < l_task: # if worker 1 is not qualified for the task
                        skills[w1][index_m] += COEF_LEARNING 

                    
                    elif w2 >= 0: # if w2 == -1 means that the task is done in solo by w1, if w2 == -2 means that the task is not assigned to any worker
                        lev_w2 = self.instance.levels_workers[w2][index_m] # level of worker 2 for the profession of the task

                        if lev_w2 < l_task: # if w2 is not qualified for the task
                            skills[w2][index_m] = skills[w2][index_m] + COEF_LEARNING


        skills = np.minimum(self.instance.levels_workers + 1, skills) # skills can not exceed the initial level + 1 for one shift
        skills = np.minimum(LEVEL_MAX, skills) # skills can not exceed the maximum level
        return skills

    def _compute_cognitive_load(self, osc, fwac, swac,level_w1, level_w2, task_done):

        """
        Calculate the cognitive load for each worker for each profession
        hypothesis: the assignment respect the rules of assignement (not 2 workers wihout level in a same task, etc...)
        """

        cognitive_load_tutors = np.zeros((self.instance.nb_workers, self.instance.nb_professions))
        cognitive_load_apprentis = np.zeros((self.instance.nb_workers, self.instance.nb_professions))
        cognitive_load_collaboration = np.zeros((self.instance.nb_workers, self.instance.nb_professions))

        for idx_op in range(len(osc)): # for each operation in the solution
            (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
            task_id_instance = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance

            if task_done[i][j] == 1 : #on calcule mental load sur les opérations faites
                
                task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
                index_m = self.instance.task_to_m[task_id] # index of the profession

                w1 = fwac[idx_op]
                w2 = swac[idx_op]


                # COLLAB
                # print(f"w1={w1}, w2={w2}, level_w1={level_w1[idx_op]}, level_w2={level_w2[idx_op]}, task_difficulty={self.instance.tasks_difficulties[task_id_instance]}")
                if level_w1[idx_op] >= self.instance.tasks_difficulties[task_id_instance] and level_w2[idx_op] >= self.instance.tasks_difficulties[task_id_instance]: # if both workers have the required level, they can collaborate
                    cognitive_load_collaboration[w1][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_COLLAB * (LEVEL_MAX + 1 - self.instance.levels_workers[w1][index_m])
                    cognitive_load_collaboration[w2][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_COLLAB * (LEVEL_MAX + 1 - self.instance.levels_workers[w2][index_m])
                    # print("collab")
                    
                # TEACHING
                elif level_w1[idx_op] >= self.instance.tasks_difficulties[task_id_instance] and level_w2[idx_op] < self.instance.tasks_difficulties[task_id_instance]: # if w1 has the required level and w2 does not, then w1 is teaching w2
                    cognitive_load_tutors[w1][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_TUTOR * (LEVEL_MAX + 1 - self.instance.levels_workers[w1][index_m])
                    cognitive_load_apprentis[w2][index_m] += self.instance.tasks_difficulties[task_id] * COEF_W_EFF + COEF_APPRENTI * (LEVEL_MAX + 1 - self.instance.levels_workers[w2][index_m])
                    # print("teaching")
                
                # ALONE NO LEVEL (pas de cognitive load pour le moment)
                # ALONE (pas de cognitive load pour le moment)

        cognitive_load = cognitive_load_tutors + cognitive_load_apprentis + cognitive_load_collaboration
        # for k in range(self.instance.nb_workers):
        #     for m in range(self.instance.nb_professions):
        #             assert cognitive_load[k,m] == excpected[k,m], f"Cognitive load is not equal for worker {k}, profession {m}, cognitive_load={cognitive_load[k,m]}, expected={excpected[k,m]}"
        # print("cognitive_load_totale=")
        # print(cognitive_load)

        # print("Total cognitive load:", np.sum(cognitive_load))
        return {"cognitive_load_totale": cognitive_load, "cognitive_load_tutors": cognitive_load_tutors, "cognitive_load_apprentis": cognitive_load_apprentis, "cognitive_load_collaboration": cognitive_load_collaboration}



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
            # print(f"operation {idx_op}: ({i}, {j})")

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
                finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][0] * PERC_SOLO_NO_LEVEL_TIME
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

    def _compute_job_done_before_limit(self, job_end_dates, job_done):
        job_done_before_limit = np.ones(self.instance.nb_jobs, dtype=int)
        
        # print("ici:", self.limit_makespan)
        if self.limit_makespan == -1 or self.limit_makespan == False:
            return job_done_before_limit * job_done
        
        for i in range(len(job_done_before_limit)):
            if job_end_dates[i] > self.limit_makespan or job_end_dates[i] == -1:
                job_done_before_limit[i] = 0

        return job_done_before_limit

    def _compute_task_not_possible_done(self, osc, fwac, swac):
        """
        Dans le vecteur osc, les tasks affecté telle que la tache qui doit etre fait avant celle la n'est pas affecté, 
        alors elles ne doivent pas être affecté
        """
        for i in range(self.instance.nb_jobs):

            idx_job_i = np.zeros(self.nb_op_in_job[i], dtype=int) # idx des ops du job i
            for j in range(self.nb_op_in_job[i]):
                    idx_job_i[j] = self.dicos.op_to_id[(i,j)]

            if np.any(fwac[idx_job_i] == -1) : # Si une des opérations du job i n'est pas affecté, alors toutes les op de ce job pas affecté
                fwac[idx_job_i] = -1
                swac[idx_job_i] = -1 # pas necessaire normalement

        return fwac, swac


    def _get_time_affectation(self, mode, start_worker_id, finish_worker_id, i, j, idx_op, instance_task_id, w1, w2, Cmax):

        finish_time_before = 0
        x_ijk = 0

        ############### INDEX TASKS MUST DONE BEFORE ##############
        constrained_operations = []
        all_op_before = np.where(self.instance.constraints_precedence_operations[i,:,j] == 1)[0]
        for p in all_op_before:
            constrained_operations.append(self.dicos.op_to_id[(i,p)]) # we add the id of the operation (i,p) to the list of constrained operations for the operation (i,j)
        # print("op=(",idx_op,")")
        # print(f"all_op_before={constrained_operations}")

        ####### ALONE #######
        if mode == 0 :

            if len(constrained_operations) > 0 :
                if any(finish_worker_id[:, constrained_operations] != -1) : # tache prec et tous affecté
                    finish_time_before = np.max(finish_worker_id[:, constrained_operations], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches

                else: #operation prec pas effectuer donc bloquer l'affectation de cette tache
                    # finish_worker_id[w1][idx_op] = -1 # deja init à -1
                    # start_worker_id[w1][idx_op] = -1 # deja init à -1
                    x_ijk = 0
                    return start_worker_id, finish_worker_id, x_ijk

            finish_by_w = np.max(finish_worker_id[w1]) # recupérer la date de fin de la dernière tâche faite par w1
            start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before)  # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
            finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][0] # calculer la date de fin de la tâche actuelle pour w1

            if finish_worker_id[w1][idx_op] > Cmax : #fin de l'operation trop tardive
                finish_worker_id[w1][idx_op] = -1
                start_worker_id[w1][idx_op] = -1
                x_ijk = 0
            else:
                x_ijk = 1



        ####### TEACHING #######
        elif mode == 1:

            if len(constrained_operations) > 0 :
                if any(finish_worker_id[:, constrained_operations] != -1) : # tache prec et tous affecté
                    finish_time_before = np.max(finish_worker_id[:, constrained_operations], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches

                else: #operation prec pas effectuer donc bloquer l'affectation de cette tache
                    # finish_worker_id[w1][idx_op] = -1 # deja init à -1
                    # start_worker_id[w1][idx_op] = -1 # deja init à -1
                    x_ijk = 0
                    return start_worker_id, finish_worker_id, x_ijk
                

            finish_by_w = max(np.max(finish_worker_id[w1]), np.max(finish_worker_id[w2])) # date de fin la moins tardive pour commencer task curr
            # print("finish_by_w=", finish_by_w)
            start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before)  # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
            start_worker_id[w2][idx_op] = start_worker_id[w1][idx_op] # commencer en même temps que w1
            finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][1]
            finish_worker_id[w2][idx_op] = start_worker_id[w2][idx_op] + self.instance.tasks_times[instance_task_id][1]

            if finish_worker_id[w1][idx_op] > Cmax : #fin de l'operation trop tardive
                finish_worker_id[w1][idx_op] = -1
                start_worker_id[w1][idx_op] = -1
                finish_worker_id[w2][idx_op] = -1
                start_worker_id[w2][idx_op] = -1
                x_ijk = 0
            else:
                x_ijk = 1

        ####### COLLABORATION #######
        elif mode == 2: # collaboration

            if len(constrained_operations) > 0 :
                if any(finish_worker_id[:, constrained_operations] != -1):
                    # print("ici")
                    finish_time_before = np.max(finish_worker_id[:, constrained_operations], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                    # print("ftb=", finish_time_before)
                # !!! Si cette tache qui doit etre fait avant n'a pas été affecté alors ne pas affecter cette tache courante !!!
                else:
                    start_worker_id[w1][idx_op] = -1
                    finish_worker_id[w1][idx_op] = -1
                    start_worker_id[w2][idx_op] = -1
                    finish_worker_id[w2][idx_op] = -1
                    x_ijk = 0
                    return start_worker_id, finish_worker_id, x_ijk

            finish_by_w = max(np.max(finish_worker_id[w1]), np.max(finish_worker_id[w2])) # date de fin la moins tardive pour commencer task curr
            # print("finish_by_w=", finish_by_w)
            start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before)  # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
            start_worker_id[w2][idx_op] = start_worker_id[w1][idx_op]
            
            finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][2]
            finish_worker_id[w2][idx_op] = finish_worker_id[w1][idx_op]
            
            if finish_worker_id[w1][idx_op] > Cmax : #fin de l'operation trop tardive
                finish_worker_id[w1][idx_op] = -1
                start_worker_id[w1][idx_op] = -1
                finish_worker_id[w2][idx_op] = -1
                start_worker_id[w2][idx_op] = -1
                x_ijk = 0
            else:
                x_ijk = 1

        ####### ALONE NO LEVEL #######
        elif mode == 3: # alone no level


            if len(constrained_operations) > 0 :
                if any(finish_worker_id[:, constrained_operations] != -1):
                    finish_time_before = np.max(finish_worker_id[:, constrained_operations], axis=0) # recupérer la date de fin de ces tâches
                    finish_time_before = np.max(finish_time_before) # date de fin la plus tardive parmi ces tâches
                # !!! Si cette tache qui doit etre fait avant n'a pas été affecté alors ne pas affecter cette tache courante !!!
                if finish_time_before == -1 : # tache qui doit etre fait avant n'a pas été affecté alors ne pas affecter cette tache courante
                    start_worker_id[w1][idx_op] = -1
                    finish_worker_id[w1][idx_op] = -1
                    x_ijk = 0
                    return start_worker_id, finish_worker_id, x_ijk
                # print("finish_time_before=", finish_time_before)

            finish_by_w = np.max(finish_worker_id[w1]) # recupérer la date de fin de la dernière tâche faite par w1
            # print("finish_by_w=", finish_by_w)
            start_worker_id[w1][idx_op] = max(finish_by_w, finish_time_before) # commencer apres la fin de ceux qui doivent etre fait avant et apres la fin des tache fait par w1
            finish_worker_id[w1][idx_op] = start_worker_id[w1][idx_op] + self.instance.tasks_times[instance_task_id][0] * PERC_SOLO_NO_LEVEL_TIME
            # print("start_worker_id=", start_worker_id[w1][idx_op], "finish_worker_id=", finish_worker_id[w1][idx_op])

            if finish_worker_id[w1][idx_op] > Cmax : #fin de l'operation trop tardive
                finish_worker_id[w1][idx_op] = -1
                start_worker_id[w1][idx_op] = -1
                x_ijk = 0
            else:
                x_ijk = 1

        return start_worker_id, finish_worker_id, x_ijk

        

    def decode_fast(self, chromosome):
        """
        Sert pour calculer rapidement la fitness d'un chromosome
        Sans calculer tous les éléments de la solution
       
        Ne modifie pas le chromosome
        """

        
        Cmax = self.constraints_config["constrained_makespan"]

        ### Copie du chromosome
        # chromosome_copy = chromosome.copy()
        # print("chromosome_copy : ")
        # print(chromosome_copy)



        # NEW VERSION SEQUENTIELLE
        ######----------------------------------------------------
        

        number_viewed_job = np.zeros(self.instance.nb_jobs, dtype=int) # pour savoir quelle op du job à étée deja vue
        # print("number of jobs=", number_viewed_job)
        # print("number of operations in each job=", self.nb_op_in_job)

        start_worker_id = np.zeros((self.instance.nb_workers, len(chromosome.osc)), dtype=float) # [w][i] = date time of task i of worker w, otherwise 0
        start_worker_id.fill(-1)
        #      0.  1.  2.  3.  4.  
        # w0   7   0   3  -1  -1  
        # w1  -1  -1   3  -1  -1  
        # w2  -1  -1  -1  -1   0

        finish_worker_id = np.zeros((self.instance.nb_workers, len(chromosome.osc)), dtype=float) # [w][i] = finish time of task i of worker w, otherwise 0
        finish_worker_id.fill(-1)

        modes = []
        lw1 = []
        lw2 = []
        job_done = np.zeros(self.instance.nb_jobs, dtype=int)
        x = np.zeros((len(job_done), self.instance.max_nb_operations, self.instance.nb_workers), dtype=int)
        z = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations, 4), dtype=int)
        d = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.nb_workers))
        f = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.nb_workers))
        is_tutor = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations, self.instance.nb_workers), dtype=int)
        task_done = np.zeros((self.instance.nb_jobs, self.instance.max_nb_operations), dtype=int)
        y = np.zeros((self.instance.nb_jobs, self.instance.nb_workers), dtype=int)
        lw1 = np.zeros((len(chromosome.osc)))
        lw2 = np.zeros((len(chromosome.osc)))
        lw1.fill(-1)
        lw2.fill(-1)

        for i in range(len(self.instance.jobs_struct)):
            # print(f"job {i} : base_worker_assignment={chromosome.base_worker_assignment[i]}")
            if chromosome.base_worker_assignment[0][i] != -1:
                y[i][chromosome.base_worker_assignment[0][i]] = 1 # y[i][w] = 1 if worker w is assigned to job i, otherwise 0


        for pos_op in range(len(chromosome.osc)): # on parcourt les opérations dans l'ordre donné par OSC
            
            
            # Savoir quelle opération du job on est en train de traiter
            i = chromosome.osc[pos_op] # job i
            j = number_viewed_job[i] # operation j of job i
            number_viewed_job[i] += 1


            # On recupère l'index de l'operation (i,j) dans le chromosome ainsi que l'index de l'op dans l'instance et les workers associés
            idx_op = self.dicos.op_to_id[(i,j)] # index of the operation (i,j) in the chromosome
            instance_task_id = self.instance.jobs_struct[i][j] # task id for know information about the task in the instance
            w1 = chromosome.fwac[idx_op]
            w2 = chromosome.swac[idx_op]

            m = self.instance.task_to_m[instance_task_id] # index of the profession



            # if w1 != -1 :
            # print(f"({i},{j}) : idx_op={idx_op}, instance_task_id={instance_task_id}, w1={w1}, w2={w2}")


            # MODE OPERATOIRE
            mode = self._get_mode_operation(instance_task_id, w1, w2)
            # print("w1=", w1, "w2=", w2, "mode=", mode, "op(", i, ",", j, ")")
            
            
            if w1 != -1 :
                lw1[idx_op] = self.instance.levels_workers[w1][m]
            else:
                lw1[idx_op] = -1

            if w2 != -1 :
                lw2[idx_op] = self.instance.levels_workers[w2][m]
            else:
                lw2[idx_op] = -1
            modes.append(mode)
            # print(f"mode operation ({i},{j}) : {mode}")
            # print("lw1=", lw1[idx_op], "lw2=", lw2[idx_op])
            # print("------")

            # START AND FINISH DATES
            
            start_worker_id, finish_worker_id, affectations = self._get_time_affectation(mode, start_worker_id, finish_worker_id, i, j, idx_op, instance_task_id, w1, w2, Cmax)
            # if w1 != -1:
            # print("op(", i, ",", j, ") : m=", mode, "s=", start_worker_id[w1,idx_op], "f=", finish_worker_id[w1,idx_op], "x=", affectations, "[w1=", w1, "w2=", w2, "]")
            #     print("w1=", w1)
            # if w2 != -1:
            #     print("w2=", w2)
            if (mode == 0 or mode == 3) and affectations == 1:
                x[i,j,w1] = affectations
                if affectations == 1:
                    z[i, j, mode] = 1
                    d[i, j, w1] = start_worker_id[w1][idx_op]
                    f[i, j, w1] = finish_worker_id[w1][idx_op]
                    task_done[i, j] = 1
             
                
            elif (mode == 1 or mode == 2) and affectations == 1:
                x[i,j,w1] = affectations
                x[i,j,w2] = affectations
                if affectations == 1:
                    z[i, j, mode] = 1
                    d[i, j, w1] = start_worker_id[w1][idx_op]
                    f[i, j, w1] = finish_worker_id[w1][idx_op]
                    d[i, j, w2] = start_worker_id[w2][idx_op]
                    f[i, j, w2] = finish_worker_id[w2][idx_op]
                    task_done[i, j] = 1
                if mode == 1:
                    is_tutor[i, j, w1] = 1
                    

            if w1 != -1 and self.nb_op_in_job[i] == number_viewed_job[i] and affectations == 1:
                job_done[i] = 1
            

        # print("start_worker_id=")
        # print(start_worker_id)
        # print("finish_worker_id=")
        # print(finish_worker_id)

        ######----------------------------------------------------
        
        # 19/08 : j'ai l'imprssion que l'on a plus besoin de cela car on peut fair le debut d'un job sans le temriner
        ### MaJ des opérations qui on pas tout leurs opérations du job affecté
        # chromosome_copy.fwac, chromosome_copy.swac =self._compute_task_not_possible_done(chromosome_copy.osc, chromosome_copy.fwac, chromosome_copy.swac)

        ### Calcul des modes et des dates de début et de fin des opérations
        # lw1, lw2, modes = self._get_modes(chromosome_copy.osc, chromosome_copy.fwac, chromosome_copy.swac)
 
        # start, finish = self._start_date_calculate(chromosome_copy.osc, chromosome_copy.fwac, chromosome_copy.swac)
        # job_done = self._compute_job_done(chromosome_copy.fwac)
        # job_end_dates = self._compute_job_end_dates(finish)
        # job_done_before_limit = self._compute_job_done_before_limit(job_end_dates, job_done)


        # x = chromosome_to_x_variable(chromosome, self.instance, self.dicos)
        # assert_base_worker_is_respcted(x, chromosome.base_worker_assignment, 1, self.instance.jobs_struct)
        

        # LES JOBS FINIT APRES LA LIMITE DE MAKESPAN NE SONT PAS AFFECTE AUX WORKERS
        # METTRE LES AFFECTATIONS A -1 POUR LES OPERATIONS DES CES JOBS
        # ET METTRE WORKER DE BASE a -1 POUR CES JOBS



        # for i in range(len(job_done_before_limit)):
        #     if job_done_before_limit[i] == 0:
        #         for j in range(len(self.instance.jobs_struct[i])):
        #             idx_op = self.dicos.op_to_id[(i,j)]
        #             chromosome_copy.fwac[idx_op] = -1
        #             chromosome_copy.swac[idx_op] = -1
        #         chromosome_copy.base_worker_assignment[0][i] = -1 # Pas de worker de base pour ce job car il n'est pas fait avant la limite de makespan
        #         chromosome_copy.base_worker_assignment[1][i] = 0 # Pas de pénalité de levels car pas affécté
        

        # lw1, lw2, modes = self._get_modes(osc, fwac, swac)
        # start, finish = self._start_date_calculate(osc, fwac, swac)
        # job_end_dates = self._compute_job_end_dates(finish)


        
        # x = chromosome_to_x_variable(chromosome_copy, self.instance, self.dicos)
        # assert_base_worker_is_respcted(x,chromosome_copy.base_worker_assignment, 1, self.instance.jobs_struct)

        

        skills = self._compute_skills(chromosome.osc, chromosome.fwac, chromosome.swac, task_done)

        # print("ici")
        # print(chromosome)
        # print(lw1)
        # print(lw2)
        # print(job_done)
        # print(task_done)
        cognitive_load_dico = self._compute_cognitive_load(chromosome.osc, chromosome.fwac, chromosome.swac, lw1, lw2, task_done)
        # print("in fast")
        # print(cognitive_load_dico)
        # print("modes=")
        # print(modes)



        # print("ici :")
        # print(x[0])
        # print("sum CL=")
        # print(np.sum(cognitive_load_dico["cognitive_load_totale"]))
        return {
            "start": start_worker_id,
            "finish": finish_worker_id,
            "skills": skills,
            "cognitive_load": cognitive_load_dico,
            "lw1": lw1,
            "lw2": lw2,
            "modes": modes,
            "job_done": job_done,
            "x":x,
            "z":z,
            "d":d,
            "f":f,
            "is_tutor": is_tutor,
            "task_done": task_done,
            "y": y,
            "Cmax": Cmax,
            # "job_end_dates": job_end_dates,
            # "job_done_before_limit": job_done_before_limit,
            # "C_max": np.max(job_end_dates),
        }

    def get_objectives(self, chromosome, ponderation_task_done="one"):

        self.decoded = self.decode_fast(chromosome)
        task_done = self.decoded["task_done"]
        obj_task_done = 0

        if ponderation_task_done == "one":
            obj_task_done = np.sum(task_done)
        elif ponderation_task_done == "difficulty":
            obj_task_done = np.sum(task_done[i,j] * self.instance.get_difficulty_and_metier_of_task(i,j)[0] for i in range(self.instance.nb_jobs) for j in range(len(self.instance.jobs_struct[i])))


        obj1 = np.sum(self.decoded["job_done"] * self.instance.resale_price_jobs) + obj_task_done
        obj2 = np.sum(self.decoded["skills"]) - np.sum(self.instance.levels_workers)
        obj3 = np.sum(self.decoded["cognitive_load"]["cognitive_load_totale"])
        return [obj1, obj2, obj3]

    def decode_solution(self, chromosome):
        """
        Sert pour calculer rapidement la fitness d'un chromosome sans calculer tous les éléments de la solution
        """        
        # print(chromosome.base_worker_assignment)

        decoded = self.decode_fast(chromosome)
        self.decoded = decoded


        makespan = np.max(decoded["finish"])


        s = Solution(self.instance) 
        # s.x = chromosome_to_x_variable(chromosome, self.instance, self.dicos)
        s.x = decoded["x"]
        s.z_auxilary = decoded["z"]
        s.l = decoded["skills"]
        s.d = decoded["d"]
        s.f = decoded["f"]
        s.is_tutor = decoded["is_tutor"]
        cognitive_load_dico = decoded["cognitive_load"]
        s.cognitive_load_total = cognitive_load_dico["cognitive_load_totale"]
        s.cognitive_load_tutors = cognitive_load_dico["cognitive_load_tutors"]
        s.cognitive_load_apprentis = cognitive_load_dico["cognitive_load_apprentis"]
        s.cognitive_load_collaboration = cognitive_load_dico["cognitive_load_collaboration"]
        # #print(np.sum(s.cognitive_load_total))

        s.job_done = decoded["job_done"]
        s.task_done = decoded["task_done"]
        s.y = decoded["y"]
        s.C_max = decoded["Cmax"]

        if chromosome.objectives is not None:
            benefit = chromosome.objectives[0]
            skills = chromosome.objectives[1]
            cognitive_load = chromosome.objectives[2]
            s.objective_values = {"0": benefit, "1": skills, "2": cognitive_load}



        # s.base_worker_assignment = base_worker_assignment
        # for i in range(self.instance.nb_jobs):
        #     if s.job_done_before_limit[i] == 0:
        #         s.base_worker_assignment[0][i] = -1
        #         s.base_worker_assignment[1][i] = -1
        
        # for idx_op in range(len(osc)):
        #     (i,j) = self.dicos.id_to_op[idx_op] # operation j of job i
        #     w1 = fwac[idx_op]
        #     w2 = swac[idx_op]
        #     mode_op = decoded["modes"][idx_op]

        #     if w1 != -1:
        #         # s.x[i,j,w1] = 1 #* s.job_done_before_limit[i]
        #         s.d[i,j,w1] = decoded["start"][w1][idx_op] #* s.job_done_before_limit[i]
        #         s.f[i,j,w1] = decoded["finish"][w1][idx_op] #* s.job_done_before_limit[i]
        #         # s.z_auxilary[i,j,mode_op] = 1 #* s.job_done_before_limit[i]
                
        #         if mode_op == 1: # teaching (tutor)
        #             s.is_tutor[i,j, w1] = 1 * s.job_done_before_limit[i]

        #     if w2 != -1:
        #         # s.x[i,j,w2] = 1 * s.job_done_before_limit[i]
                  # s.d[i,j,w2] = decoded["start"][w2][idx_op] * s.job_done_before_limit[i]
        #         s.f[i,j,w2] = decoded["finish"][w2][idx_op] * s.job_done_before_limit[i]
        #         # s.z_auxilary[i,j,mode_op] = 1 * s.job_done_before_limit[i]


        # #print("solution decoded : ")
        # print(s.x[0,:,:])
        return s

########################################
########## Fonction de voisinage
########################################

class Neighborhood:
    """ Classe de base pour tous les voisinages. Chaque sous-classe implémente generate() """
    def generate(self, chromosome):
        pass


# ==============================================================================
# VOISINAGES SUR L'ORDRE DE SÉQUENCEMENT (osc)
# Impact principal : benefit (quels jobs finissent avant Cmax)
# ==============================================================================

class SwapOperationNeighborhood(Neighborhood):
    """
    [STOCHASTIQUE] Permute aléatoirement 2 positions dans osc
    Génère n voisins indépendants
    Ne change qu'une paire de positions à la fois
    """
    def __init__(self, n, nb_job_no_skills=0):
        self.n = n # nombre de voisins à générer
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        voisins = []
        # tire n paires de positions aléatoires
        o1 = np.random.choice(len(chromosome.osc), size=self.n, replace=True)
        o2 = np.random.choice(len(chromosome.osc), size=self.n, replace=True)
        for idx_1, idx_2 in zip(o1, o2):
            v = chromosome.copy()
            v.osc[idx_1], v.osc[idx_2] = v.osc[idx_2], v.osc[idx_1]
            voisins.append(v)
        return voisins

class MoveOneOperationNeighborhood(Neighborhood):
    """
    [Mi-DETERMINISTE] Choisit une position aléatoire dans osc et la permute avec
    toutes les autres positions possibles (d'un job différent)
    Stochastique dans le choix de la position source
    déterministe dans les destinations
    Explore toutes les insertions d'une opération.
    """
    def __init__(self, nb_job_no_skills=0):
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        voisins = []
        idx = np.random.choice(len(chromosome.osc))  # position source choisie aléatoirement

        for pos in range(len(chromosome.osc)):
            # on ne permute pas avec une position du même job (permutation triviale)
            if chromosome.osc[pos] != chromosome.osc[idx]:
                v = chromosome.copy()
                v.osc[pos], v.osc[idx] = v.osc[idx], v.osc[pos]
                voisins.append(v)
        return voisins

class SwapFixedOperationsNeighborhood(Neighborhood):
    """
    [STOCHASTIQUE] Fixe k positions aléatoires dans osc et permute aléatoirement les positions restantes
    Répété n fois pour générer n voisins
    """
    def __init__(self, k, n, nb_job_no_skills=0):
        self.k = k # nombre de positions fixées
        self.n = n # nombre de voisins à générer
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
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

class MoveJobNeighborhood(Neighborhood):
    """
    [STOCHASTIQUE]
    Choisit un job aléatoirement, retire toutes ses occurrences de OSC,
    puis les réinsère à des positions aléatoires.

    Génère n voisins indépendants.
    """

    def __init__(self, n, nb_job_no_skills=0):
        self.n = n
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        voisins = []

        # Choisir un job aléatoirement
        job = np.random.choice(np.unique(chromosome.osc))

        # Nombre d'opérations du job
        nb_operations = np.sum(chromosome.osc == job)

        # Retirer toutes les occurrences du job
        osc_without_job = chromosome.osc[chromosome.osc != job]

        for _ in range(self.n):

            # Positions des opérations du job dans le nouvel OSC
            positions = np.random.choice(len(chromosome.osc), size=nb_operations, replace=False)
            positions = np.sort(positions)

            # Construction du nouvel OSC
            new_osc = np.empty(len(chromosome.osc), dtype=chromosome.osc.dtype)

            new_osc[positions] = job

            mask = np.ones(len(chromosome.osc), dtype=bool)
            mask[positions] = False

            new_osc[mask] = osc_without_job

            # Création du voisin
            v = chromosome.copy()
            v.osc = new_osc

            voisins.append(v)

        return voisins


class ShuffleOperationsNeighborhood(Neighborhood):
    """
    [STOCHASTIQUE] Permutation aléatoire complète de osc. Génère n voisins
    Perturbation forte : repart presque de zéro sur l'ordonnancement
    Utile comme voisinage de perturbation dans le VNS pour sortir de maxima locaux
    """
    def __init__(self, n, nb_job_no_skills=0):
        self.n = n # nombre de voisins à générer
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        voisins = []
        for _ in range(self.n):
            v = chromosome.copy()
            v.osc = np.random.permutation(v.osc)
            voisins.append(v)
        return voisins


# ==============================================================================
# VOISINAGES SUR L'AFFECTATION DES WORKERS (fwac / swac)
# ==============================================================================

class MoveSecondWorkerNeighborhood(Neighborhood):
    """
    [DÉTERMINISTE sur les workers, STOCHASTIQUE sur le choix de l'opération]
    Choisit une opération aléatoirement et teste tous les workers possibles pour swac
    swac est le 2e worker (apprenti ou collaborateur ou -1)
    -1 est toujours inclus comme option (solo sans 2e worker)

    Contraintes respectées :
        - si le worker de base du job est dans swac (il est apprenti), on ne peut pas changer swac -> retourne voisinage vide
        - si fwac n'a pas le niveau requis alors swac ne change pas
    """
    def __init__(self, generator, nb_job_no_skills=0):
        self.generator = generator
        self.nb_job_no_skills = nb_job_no_skills

    def _get_possible_workers_for_operation(self, idx_op):
        """ Workers éligibles pour swac : niveau au plus -1 par rapport à la tâche, + option solo (-1) """
        (i, j) = self.generator.dicos.id_to_op[idx_op]
        # seuls les workers à au plus 1 niveau en dessous sont éligibles comme apprentis
        idx_workers = self.generator.get_worker_at_most_1_level_for_operation_and_not_qualified((i, j))
        idx_workers = np.append(idx_workers, -1)  # -1 = solo, pas de 2e worker
        return idx_workers

    def generate(self, chromosome):
        voisins = []
        idx = np.random.randint(len(chromosome.osc))
        (i, j) = self.generator.dicos.id_to_op[idx]

        # si le worker de base du job i est dans swac (il est apprenti car il n'a pas le niveau),
        # changer swac le retirerait de l'opération -> violation de la contrainte worker de base
        base_worker_i = chromosome.base_worker_assignment[0][i]
        if base_worker_i >= 0 and chromosome.fwac[idx] != base_worker_i:
            return voisins

        # si fwac n'a pas le niveau requis (mode solo sans niveau), swac doit rester -1
        task_id = self.generator.instance.jobs_struct[i][j]
        m = self.generator.instance.task_to_m[task_id]
        l_task = self.generator.instance.tasks_difficulties[task_id]
        w1 = chromosome.fwac[idx]
        if w1 == -1 or self.generator.instance.levels_workers[w1][m] < l_task:
            return voisins

        workers_possible = self._get_possible_workers_for_operation(idx)
        for w in workers_possible:
            if w != chromosome.swac[idx]:  # on ne génère pas le chromosome identique
                v = chromosome.copy()
                v.swac[idx] = w
                voisins.append(v)
        v = chromosome.copy()
        v.swac[idx] = -1  # si le worker fait seul aussi
        voisins.append(v)
        return voisins

class ChangebaseWorkerNeighborhood(Neighborhood):
    """
    [DÉTERMINISTE] Change le worker de base d'un job en respectant sa catégorie :
        - catégorie 0 (sans pénalité) : le nouveau worker de base doit aussi être sans pénalité
        - catégorie 1 (avec pénalité, solo sans niveau autorisé) : reste avec pénalité

    Après avoir changé le worker de base, fwac et swac sont recalculés via
    assign_fwac_swac_with_base_worker pour rester cohérents
    Explore tous les jobs dans un ordre aléatoire, génère 1 voisin par alternative trouvée
    """
    def __init__(self, nb_job_no_skills, generator, verbose=False):
        self.generator = generator
        self.nb_job_no_skills = nb_job_no_skills
        self.verbose = verbose

    def _find_other_base_worker_for_job(self, chromosome, indx_job):
        """
        Pour chaque job, génère toutes les affectations de worker de base alternatives
        en restant dans la même catégorie (avec ou sans pénalité).
        Retourne une liste de matrices base_worker_assignment candidates.
        """
        voisins_base_workers = []

        #Recuperer les workers ayant un level gap d'au moins 1 avec la premiere op de chaque job

        level_gap = self.generator.calculate_matrix_difference_level_tasks_workers()
        worker_index_available_base_worker = level_gap[:,0] >=-1
        worker_index_available_base_worker = np.where(worker_index_available_base_worker[indx_job])[0]

        if len(worker_index_available_base_worker) == 0:
            return voisins_base_workers  # pas d'alternatives pour ce job
        # print("worker_index_available_base_worker for this job : ", worker_index_available_base_worker)

        for w in worker_index_available_base_worker :
            if w != chromosome.base_worker_assignment[0][indx_job]:  # on ne génère pas le chromosome identique
                base_worker_assignment_new = np.copy(chromosome.base_worker_assignment)
                base_worker_assignment_new[0][indx_job] = w
                voisins_base_workers.append(base_worker_assignment_new)

        return voisins_base_workers

    def generate(self, chromosome):
        """
        Génère des voisins en changeant le worker de base d'un job.
        fwac et swac sont recalculés pour chaque nouveau worker de base.
        """
        
        indx_job = np.random.randint(0, self.generator.instance.nb_jobs)
        # print("indx_job : ", indx_job)

        liste_voisins_base_workers = self._find_other_base_worker_for_job(chromosome, indx_job)
        voisins = []

        for base_worker_assignment in liste_voisins_base_workers:
            v = chromosome.copy()
            # recalcul de fwac et swac cohérents avec le nouveau worker de base
            v.fwac, v.swac = self.generator.assign_fwac_swac_with_base_worker(base_worker_assignment, v.osc)
            v.base_worker_assignment = np.copy(base_worker_assignment)
            voisins.append(v)

        return voisins


# ==============================================================================
# VOISINAGES COMPOSÉS
# ==============================================================================

class multipleNeighborhood_cumulative(Neighborhood):
    """
    Applique n1 sur le chromosome courant, puis n2 sur chacun des voisins de n1
    Retourne l'union des voisins de n1 et des voisins de n2
    Explore des solutions à 2 pas de distance
    """
    def __init__(self, n1, n2, nb_job_no_skills=0):
        self.n1 = n1
        self.n2 = n2
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        v1 = self.n1.generate(chromosome)
        v2 = []
        for v in v1:
            v2 += self.n2.generate(v)  # applique n2 sur chaque voisin de n1
        return v1 + v2

class multipleNeighborhood_alternating(Neighborhood):
    """
    Choisit aléatoirement (50/50) entre n1 et n2 à chaque appel
    Permet de diversifier les mouvements sans les cumuler
    """
    def __init__(self, n1, n2, nb_job_no_skills=0):
        self.n1 = n1
        self.n2 = n2
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        if np.random.rand() < 0.5:
            return self.n1.generate(chromosome)
        else:
            return self.n2.generate(chromosome)

class RandomRestartNeighborhood(Neighborhood):
    """
    [RESTART] Génère n solutions aléatoires pour redémarrer la recherche
    Modifie toutes les composante du chromosome
    """

    def __init__(self, generator, n=5, nb_job_no_skills=0):
        self.generator = generator
        self.n = n
        self.nb_job_no_skills = nb_job_no_skills

    def generate(self, chromosome):
        return [ self.generator.generate_random(nb_job_no_skills=self.nb_job_no_skills) for _ in range(self.n) ]


# class PrioritizeUnfinishedJobNeighborhood(Neighborhood):
#     """
#     [GUIDÉ] Identifie les jobs non terminés (job_done=0) et déplace leurs opérations
#     vers le début de OSC pour maximiser leur chance de terminer avant Cmax.

#     Le job cible est tiré aléatoirement parmi les non-terminés, pondéré par prix de revente.
#     Les opérations du job sont insérées dans le premier quart de OSC.

#     Requiert chromosome.job_done renseigné (disponible après evaluate_agg).
#     Cible directement l'objectif dominant (benefit × 100).
#     """

#     def __init__(self, instance, n=5, nb_job_no_skills=0):
#         self.instance = instance
#         self.n = n
#         self.nb_job_no_skills = nb_job_no_skills

#     def generate(self, chromosome):
#         voisins = []

#         if chromosome.job_done is None:
#             return voisins

#         unfinished = np.where(chromosome.job_done == 0)[0]
#         if len(unfinished) == 0:
#             return voisins

#         # Tirage pondéré par prix de revente : favorise les jobs les plus rentables
#         prices = self.instance.resale_price_jobs[unfinished].astype(float)
#         total = prices.sum()
#         probs = prices / total if total > 0 else None
#         job = np.random.choice(unfinished, p=probs)

#         nb_op = int(np.sum(chromosome.osc == job))
#         osc_without_job = chromosome.osc[chromosome.osc != job]

#         # Zone d'insertion : premier quart de OSC (au moins nb_op positions)
#         max_insert = max(nb_op, len(chromosome.osc) // 4)

#         for _ in range(self.n):
#             positions = np.random.choice(max_insert, size=nb_op, replace=False)
#             positions = np.sort(positions)

#             new_osc = np.empty(len(chromosome.osc), dtype=chromosome.osc.dtype)
#             new_osc[positions] = job

#             mask = np.ones(len(chromosome.osc), dtype=bool)
#             mask[positions] = False
#             new_osc[mask] = osc_without_job

#             v = chromosome.copy()
#             v.osc = new_osc
#             voisins.append(v)

#         return voisins


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
        self.all_viewed = None
        self.best_list = None
        self.i_find_best_list = None
    
    def run(self, max_evaluation, initial_chromosome=None, verbose=False):
        """
        Effectue une recherche aléatoire sur l'espace des solutions
        :param max_evaluation: nombre maximum d'évaluations à effectuer
        :param initial_chromosome: chromosome initial à partir duquel commencer la recherche (optionnel)
        Returns:
            all_viewed_list: liste de tous les chromosomes évalués
            best_list: liste des meilleurs chromosomes trouvés au fil des itérations
            i_find_best_list: liste des itérations où un nouveau meilleur chromosome a été trouvé
            best_list[-1]: le meilleur chromosome trouvé à la fin de la recherche
        """

        self.evaluator.reset_evaluations()
        best_list = []
        all_viewed_list = []
        i_find_best_list = [] # itération où l'on a trouvé le meilleur chromosome jusqu'à présent 
        best_fitness = -float('inf')

        for i in tqdm.trange(max_evaluation, desc="Random Search Iterations"): # 1 appel a evaluate par boucle donc ok 
            if initial_chromosome is not None:
                chromosome : Chromosome = initial_chromosome.copy()
            else:
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

        self.all_viewed = all_viewed_list
        self.best_list = best_list
        self.i_find_best_list = i_find_best_list

        return all_viewed_list, best_list, i_find_best_list, best_list[-1]

    def plot_each_objectives(self):
        """ 
        Affiche l'évolution de chaque objectif au fil des itérations pour les meilleurs chromosomes trouvés
        3 graphiques dans une seule figure : objectif 1, objectif 2, objectif 3
        """
        if self.best_list is not None:
            x = np.zeros((len(self.best_list), 3))

            for i, chromosome in enumerate(self.best_list):
                self.evaluator.evaluate_agg(chromosome)
                for j in range(3):
                    x[i, j] = chromosome.objectives[j]

            # indice des solutions
            solutions = np.arange(len(self.best_list))

            # 3 graphiques dans une seule figure
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))

            for j in range(3):
                axes[j].plot(solutions, x[:, j], marker='o')
                axes[j].set_xlabel("Solution")
                axes[j].set_ylabel(f"Objective {j + 1}")
                axes[j].set_title(f"Evolution of objective {j + 1}")
                axes[j].grid(True)

            plt.tight_layout()
            plt.show()

class LocalSearch:

    def __init__(self, generator, evaluator, neighborhood, selector):
        self.generator = generator
        self.evaluator = evaluator
        self.neighborhood = neighborhood
        self.selector = selector
        self.constraints_config = self.generator.constraints_config

    def run(self, max_evaluation, patience=None, verbose=True, initial_chromosome=None):
        """
        Recherche locale avec un voisinage donné
        :param max_evaluation: nombre maximum d'évaluations à effectuer
        :param patience: nombre d'évaluations consécutives sans amélioration avant d'arreter la recherche (optionnel)
        :param verbose: si True, affiche les informations sur l'évolution de la recherche
        :param initial_chromosome: chromosome initial à partir duquel commencer la recherche (optionnel)
        Returns:
            all_viewed_list: liste de tous les chromosomes évalués
            best_list: liste des meilleurs chromosomes trouvés au fil des itérations
            best_eval_list: liste des évaluations où un nouveau meilleur chromosome a été trouvé
        """

        not_found_best = 0

        start_eval = self.evaluator.nb_evaluations
        target_eval = start_eval + max_evaluation

        ##############################
        ###### Solution initiale ######
        ##############################

        if initial_chromosome is not None:
            chromosome = initial_chromosome.copy()
        else:
            chromosome = self.generator.generate_random(nb_job_no_skills=self.neighborhood.nb_job_no_skills)

        # Evaluation chromosome initiale
        chromosome.fitness = self.evaluator.evaluate_agg(chromosome)

        if verbose:
            print(f"Initial fitness: {chromosome.fitness}")
            print(f"Initial objectives: {chromosome.objectives}")

        best_list = [chromosome]
        all_viewed_list = [chromosome]
        best_eval_list = [self.evaluator.nb_evaluations]

        # Dernière évaluation où une amélioration a été trouvée
        last_improvement_eval = self.evaluator.nb_evaluations


        ##############################
        ###### Recherche locale ######
        ##############################
        
        while self.evaluator.nb_evaluations < target_eval:

            voisins = self.neighborhood.generate(chromosome)

            # Voisinage stochastique :
            # si aucun voisin n'est généré, on peut réessayer au prochain appel
            if len(voisins) == 0:
                not_found_best += 1
                continue

            # Budget restant
            restant_eval = target_eval - self.evaluator.nb_evaluations
            voisins = voisins[:restant_eval]


            if len(voisins) == 0:
                not_found_best += 1
                # Si plus de budget pour évaluer les voisins, on arrête la recherche locale
                break

            eval_before = self.evaluator.nb_evaluations

            fitness_v = np.zeros(len(voisins))
            for i in range(len(voisins)):
                fitness_v[i] = self.evaluator.evaluate_agg(voisins[i])

            selected_idx = self.selector.select(fitness_v)

            if selected_idx is None:
                break

            selected = voisins[selected_idx]

            # Evaluation exacte à laquelle ce voisin a été évalué
            selected_eval = eval_before + selected_idx + 1

            #######################
            ##### Acceptation #####
            #######################

            if selected.fitness >= chromosome.fitness:
                chromosome = selected
                best_list.append(chromosome)
                best_eval_list.append(selected_eval)
                last_improvement_eval = selected_eval
                if verbose :
                    print(f"Evaluation {selected_eval}: New best fitness: {chromosome.fitness}")
                if selected.fitness == chromosome.fitness:
                    not_found_best = 0
                else : 
                    not_found_best += 1
            else:
                not_found_best += 1
            all_viewed_list.append(chromosome)

            ###################
            ##### Patience ####
            ###################

            if (patience is not None) and (not_found_best >= patience):
                if verbose:
                    print(f"Stopping: {patience} evaluations without improvement.")
                break

        return all_viewed_list, best_list, best_eval_list

class MultiNeighborhoodBenchmark:
    """
    Lance une LocalSearch indépendante par voisinage fourni et compare leurs performances.

    Utilisation :
        benchmark = MultiNeighborhoodBenchmark(
            generator, evaluator,
            neighborhoods=[
                (SwapOperationNeighborhood(n=5), "Swap OSC"),
                (MoveSecondWorkerNeighborhood(generator), "Move SWAC"),
            ]
        )
        benchmark.run(max_iter=100)
        benchmark.plot_fitness_comparison()
    """

    def __init__(self, generator, evaluator, neighborhoods, selector=None):
        """
        neighborhoods : liste de tuples (Neighborhood, str) — (voisinage, label affiché)
        selector      : Selector commun à toutes les recherches (BestSelector par défaut)
        """
        self.generator: ChromosomeGenerator = generator
        self.evaluator: Evaluator = evaluator
        self.neighborhoods = neighborhoods
        self.selector = selector if selector is not None else BestSelector()
        self.results = {}

    def run(self, max_evaluations, verbose=False, shared_start=False):
        """
        Lance une LocalSearch par voisinage.
        shared_start=True  : tous les voisinages partent du même chromosome initial (comparaison équitable).
        shared_start=False : chaque voisinage part d'un chromosome aléatoire indépendant.
        """
        initial_chromosome = None
        if shared_start:
            nb_job_no_skills = self.neighborhoods[0][0].nb_job_no_skills
            initial_chromosome = self.generator.generate_random(nb_job_no_skills=nb_job_no_skills)
            if verbose:
                print(f"Point de départ commun — fitness initiale : {self.evaluator.evaluate_agg(initial_chromosome):.4f}")

        for neighborhood, label in self.neighborhoods:
            if verbose:
                print(f"\n=== Voisinage : {label} ===")
            ls = LocalSearch(self.generator, self.evaluator, neighborhood, self.selector)
            all_viewed, best_list, best_eval_list = ls.run(max_evaluations, verbose=verbose, initial_chromosome=initial_chromosome)
            self.results[label] = {
                "all_viewed": all_viewed,
                "best_list": best_list,
                "best_fitness": best_list[-1].fitness if best_list else None,
                "best_evaluations": best_eval_list,
            }
        return self.results

    def plot_fitness_comparison(self, render="html", save_path=None):
        """
        Trace l'évolution de la fitness à chaque itération pour chaque voisinage.
        Les étoiles indiquent les itérations où la meilleure fitness a été améliorée.
        """
        import plotly.graph_objects as go
        import plotly.express as px

        palette = px.colors.qualitative.Plotly
        fig = go.Figure()

        for idx, (label, res) in enumerate(self.results.items()):
            color = palette[idx % len(palette)]
            fitnesses = [c.fitness for c in res["all_viewed"]]
            iterations = list(range(len(fitnesses)))

            # Courbe principale : fitness courante à chaque itération
            fig.add_trace(go.Scatter(
                x=iterations,
                y=fitnesses,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))

            # Étoiles aux itérations où la meilleure fitness globale a été améliorée
            improvement_iters = [
                i for i in range(1, len(fitnesses))
                if fitnesses[i] > fitnesses[i - 1]
            ]
            if improvement_iters:
                fig.add_trace(go.Scatter(
                    x=improvement_iters,
                    y=[fitnesses[i] for i in improvement_iters],
                    mode="markers",
                    marker=dict(size=10, symbol="star", color=color),
                    name=f"{label} — améliorations",
                    showlegend=True,
                ))

        fig.update_layout(
            title="Comparaison des voisinages — évolution de la fitness",
            xaxis_title="Itération",
            yaxis_title="Fitness",
            legend_title="Voisinage",
            template="plotly_white",
            hovermode="x unified",
        )

        path = save_path if save_path else "../results/neighborhood_fitness_comparison.html"
        fig.write_html(path)
        fig.show()
        return fig

    def validate_results(self, decoder, instance, constraints_config, validate_fn):
        """
        Pour chaque voisinage, décode le meilleur chromosome trouvé et le valide avec validate_fn.

        Paramètres :
            decoder          : instance de Decoder
            instance         : instance du problème
            constraints_config : dict de configuration des contraintes
            validate_fn      : la fonction validate_model_solution importée depuis utils

        Exemple d'appel :
            benchmark.validate_results(dec, instance, constraints_config, validate_model_solution)
        """
        import pandas as pd
        from IPython.display import display

        for label, res in self.results.items():
            best_chromosome = res["best_list"][-1] if res["best_list"] else None
            
            if best_chromosome is None:
                print(f"[{label}] Aucun chromosome disponible.")
                continue
            
            # print("nb_chromosomes=", len(best_chromosome))
            print(f"\n{'='*60}")
            print(f"Voisinage : {label}  |  fitness : {best_chromosome.fitness:.4f}")
            print(f"{'='*60}")

            s = decoder.decode_solution(best_chromosome)
            row, ok = validate_fn(
                instance=instance,
                solution=s,
                constraints_config=constraints_config,
            )
            display(pd.DataFrame([row]))
            if not ok:
                print(f"  ⚠️  Solution INVALIDE pour le voisinage '{label}'")

class MultipleLocalSearch:
    def __init__(self, generator, evaluator, neighborhood, selector=None):
        self.generator    = generator
        self.evaluator    = evaluator
        self.neighborhood = neighborhood
        self.selector     = selector if selector is not None else BestSelector()

        self.all_viewed    = []
        self.best_list     = []
        self.best_eval_list = []
        self.best          = None

    def run(self, max_evaluations, patience, verbose=False):
        """
        Fait une recherche local multi start avec le voisinage donné jusqu'à épuisement du budget d'évaluations.
        Paramètres :
            max_evaluations : budget total d'évaluations
            patience        : patience passée à la LS interne (évaluations sans amélioration)
            verbose         : si True, affiche les informations sur l'évolution de la recherche
        Returns:
            all_viewed      : tous les chromosomes visités
            best_list       : toutes les améliorations trouvées (n'est pas forcément strict croissant car meilleure trouvé a présent pour chaque restart)
            best_eval_list  : évaluation absolue de chaque amélioration (même taille que best_list)
            best            : meilleur chromosome global
        """
        self.all_viewed     = []
        self.best_list      = []
        self.best_eval_list = []
        self.best           = None
        self.evaluator.reset_evaluations()

        while self.evaluator.nb_evaluations < max_evaluations:
            restant = max_evaluations - self.evaluator.nb_evaluations
            ls = LocalSearch(self.generator, self.evaluator, self.neighborhood, self.selector)
            av, best_restart, best_eval_restart = ls.run(restant, patience=patience, verbose=False)

            self.all_viewed.extend(av)

            if self.best is None or best_restart[-1].fitness > self.best.fitness:
                self.best = best_restart[-1]
                self.best_list.extend(best_restart)
                self.best_eval_list.extend(best_eval_restart)

            if verbose:
                print(f"restart {len(self.best_list)} | fitness={best_restart[-1].fitness:.4f} | best={self.best.fitness:.4f} | evals={self.evaluator.nb_evaluations}/{max_evaluations}")

        return self.all_viewed, self.best_list, self.best_eval_list, self.best

class VNS:
    """
    Variable Neighborhood Search (VNS)

    À chaque itération externe :
        k = 0
        Tant que k < nb_voisinages et budget non épuisé :
            1. Shaking    : tire un voisin aléatoire dans le voisinage k
            2. Local search : améliore ce voisin avec le voisinage 0 (le plus fin)
            3. Acceptation  :
                - si le candidat est meilleur → on se déplace et on repart de k=0
                - sinon → k+1

    Paramètres :
        neighborhoods : liste ordonnée du plus petit au plus grand voisinage
                        neighborhoods[0] est utilisé pour la LS interne
        ls_patience   : patience passée à la LS interne (évaluations sans amélioration)
    """

    def __init__(self, generator, evaluator, ls_neighborhood, neighborhoods, selector=None, ls_patience=100):
        self.generator    = generator
        self.evaluator    = evaluator
        self.ls_neighborhood = ls_neighborhood 
        self.neighborhoods = neighborhoods
        self.selector     = selector if selector is not None else BestSelector()
        self.ls_patience  = ls_patience

        self.all_viewed     = []
        self.best_list      = []
        self.best_eval_list = []
        self.best           = None

    def _local_search(self, initial_chromosome, restant):
        """LS interne depuis initial_chromosome avec le budget restant."""
        ls = LocalSearch(self.generator, self.evaluator, self.ls_neighborhood[0], self.selector)
        av, best_restart, best_eval_restart = ls.run(
            restant,
            patience=self.ls_patience,
            verbose=False,
            initial_chromosome=initial_chromosome,
        )
        return av, best_restart, best_eval_restart

    def run(self, max_evaluations, init_chromosome=None, verbose=False):
        """
        Paramètres :
            max_evaluations : budget total d'évaluations
        Retourne :
            all_viewed      : tous les chromosomes visités
            best_list       : toutes les améliorations trouvées
            best_eval_list  : évaluation absolue de chaque amélioration (même taille que best_list)
            best            : meilleur chromosome global
        """
        self.all_viewed     = []
        self.best_list      = []
        self.best_eval_list = []
        self.best           = None
        self.evaluator.reset_evaluations()

        # solution initiale
        nb_job_no_skills = self.generator.constraints_config.get("nb_job_no_skills", 0)

        if init_chromosome is not None:
            current = init_chromosome
        else: 
            current = self.generator.generate_random(nb_job_no_skills=nb_job_no_skills)
        self.evaluator.evaluate_agg(current)
        self.all_viewed.append(current)
        self.best_list.append(current)
        self.best_eval_list.append(self.evaluator.nb_evaluations)
        self.best = current

        nb_appel_voisinage = [0] * len(self.neighborhoods)


        if verbose:
            print(f"Fitness initiale : {current.fitness:.4f}")

        while self.evaluator.nb_evaluations < max_evaluations:

            k = 0
            while k < len(self.neighborhoods) and self.evaluator.nb_evaluations < max_evaluations:

                voisins = self.neighborhoods[k][0].generate(current)
                nb_appel_voisinage[k] += 1
                if len(voisins) == 0:# and k < len(self.neighborhoods) - 1:
                    k += 1
                    continue

                # shaking : 1 évaluation consommée
                shaken = voisins[np.random.randint(len(voisins))]
                self.evaluator.evaluate_agg(shaken)

                # LS interne avec le budget restant
                restant = max_evaluations - self.evaluator.nb_evaluations
                budget_ls = min(restant, self.ls_patience)
                # print("apl ls avec restant=", restant, "eval=", self.evaluator.nb_evaluations, "k=", k, "patience=", self.ls_patience, "best=", self.best.fitness)

                av, best_restart, best_eval_restart = self._local_search(shaken, budget_ls)
                self.all_viewed.extend(av)
                candidate = best_restart[-1]

                # accumule les améliorations trouvées par la LS interne
                if candidate.fitness > current.fitness:
                    self.best_list.extend(best_restart)
                    self.best_eval_list.extend(best_eval_restart)
                    current = candidate
     
                    k = 0 # on repasse au voisinage le plus fin
                    if candidate.fitness >= self.best.fitness:
                        self.best = candidate.copy()
                    # print(f"Amélioration | fitness={current.fitness:.4f} | evals={self.evaluator.nb_evaluations}/{max_evaluations}", "k=", k)
                    if verbose:
                        print(f"Amélioration | fitness={current.fitness:.4f} | evals={self.evaluator.nb_evaluations}/{max_evaluations}", "k=", k)
                else:
                    # current = candidate
                    if k == len(self.neighborhoods) - 1:
                        # print("Re - Run avec candidat moins performant")
                        # current = candidate
                        current = self.best.copy() # run a partir du meme candidat
                    k += 1

        print("\nNombre d'appels par voisinage :")
        print(nb_appel_voisinage)         
        print("best =", self.best.fitness, "objectives=", self.best.objectives, "evals=", self.evaluator.nb_evaluations)
        return self.all_viewed, self.best_list, self.best_eval_list, self.best