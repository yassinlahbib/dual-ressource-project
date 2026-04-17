import numpy as np
import plotly.figure_factory as ff
import pandas as pd
import hashlib # for generating colors for the Gantt chart based on sub-operation index
import networkx as nx
import matplotlib.pyplot as plt

def read_file(file_path):
    file = open(file_path, "r")

    nb_sub_operations = 0
    res = dict()
    
    for line in file:

        if line.strip() == "<number of jobs>":
            nb_jobs = int(file.readline().strip())
            res["nb_jobs"] = nb_jobs


        elif line.strip() == "<number of professions>":
            nb_professions = int(file.readline().strip())
            res["nb_professions"] = nb_professions


        elif line.strip() == "<professions detailed>":
            line = file.readline().strip().split(" ")
            assert nb_professions == len(line), "Le nombre de corps de métier doit être égale à la taille du vecteur professions detailed"
            nb_sub_operations_profession = np.zeros(nb_professions)

            for s in range(nb_professions):
                nb_sub_operations_profession[s] = int(line[s])
            res["nb_sub_operations_profession"] = nb_sub_operations_profession
            # print("nb_sub_operations_profession=", nb_sub_operations_profession)
            nb_sub_operations = int(np.sum(nb_sub_operations_profession))
            res["nb_sub_operations"] = nb_sub_operations
            # print("nb_sub_operations=", nb_sub_operations)


        elif line.strip() == "<sub_op(difficulty and times)>":
            # print("here !!!!!!!!!!!!!!!")
            dict_sub_op_to_m = dict() 
            sub_operations_difficulties = []
            sub_operations_times = np.zeros((nb_sub_operations, 3)) # 3 columns for doing alone, with learning or collaboratively
            index_sub_op = 0



            # print("AVANT WHILE line =", line)
            while line != '': # fin de lecture de la section
                line = file.readline().strip()
                if line == "": # fin de lecture de la section
                    break
                # print("line=", line)
                if line[0] != "m": # read the profession index 
                    # print("line[0]=", line[0])
                    line = line.split(" ")
                    sub_operations_difficulties.append(int(line[0])) # difficulty of sub-operation s
                    sub_operations_times[index_sub_op][0] = float(line[1]) # processing time of sub-operation s if done alone
                    sub_operations_times[index_sub_op][1] = float(line[2]) # processing time of sub-operation s if done with learning effect
                    sub_operations_times[index_sub_op][2] = float(line[3]) # processing time of sub-operation s if done collaboratively
                    dict_sub_op_to_m[index_sub_op] = m
                    # print("lineFIN =", line)
                    index_sub_op += 1
                if line[0] == "m":
                    m = int(line.split("m")[1]) -1
            res["sub_operations_difficulties"] = np.array(sub_operations_difficulties)
            res["sub_operations_times"] = sub_operations_times

            # print("--------------------------------")
            # print("sub_operations_difficulties=")
            # print(sub_operations_difficulties)
            # print("--------------------------------")
            # print("sub_operations_times=")
            # print(sub_operations_times)
            # print("--------------------------------")

            # print("dict_sub_op_to_m=", dict_sub_op_to_m)
            res["dict_sub_op_to_m"] = dict_sub_op_to_m



        elif line.strip() == "<maximal number of operations>":
            max_nb_operations = int(file.readline().strip())
            res["max_nb_operations"] = max_nb_operations

        
        elif line.strip() == "<maximal number of sub-operations per operation>":
            max_nb_sub_operations = int(file.readline().strip())
            res["max_nb_sub_operations"] = max_nb_sub_operations
            constraints_precedence_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_operations)) # consideration que le nombre d'operations par job est de même ordre de grandeur pour faire matrice d'adjacence des contraintes de precedences
            constraints_precedence_sub_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_sub_operations, max_nb_sub_operations))




        elif line.strip() == "<number of workers>":
            nb_workers = int(file.readline().strip())
            res["nb_workers"] = nb_workers
        
        elif line.strip() == "<levels workers>":
            levels_workers = np.zeros((nb_workers, nb_professions))
            for i in range(nb_workers):
                line = file.readline().strip().split(" ")
                for j in range(nb_professions):
                    levels_workers[i][j] = int(line[j])

            res["levels_workers"] = levels_workers
            # print("levels_workers=", levels_workers) 


        elif line.strip() == "<difficulty of jobs>":
            difficulty_jobs = np.zeros(nb_jobs)
            line = file.readline().strip().split(" ")
            for i in range(nb_jobs):
                difficulty_jobs[i] = int(line[i])
            # print("difficulty_jobs=", difficulty_jobs)
            res["difficulty_jobs"] = difficulty_jobs

        
        elif line.strip() == "<jobs>":
            jobs_struct = [ [] for _ in range(nb_jobs) ]
            job_index = 0
            
            while job_index <= nb_jobs :
                # print("job_index=", job_index)
                line = file.readline().strip()
                if line == "": # fin de lecture de la section
                    break
                if line[0] == "J":
                    job_index += 1
                    operation_index = 0
                else:
                    operation = line.split(" ") # 1 4 5
                    jobs_struct[job_index-1].append([]) # pour ajouter les sous opérations de l'opération courante à la structure du job courant
                    for sub_op in range(len(operation)):
                        jobs_struct[job_index-1][operation_index].append(int(operation[sub_op])-1)
                    operation_index += 1
            res["jobs_struct"] = jobs_struct

            # print("jobs_struct=")
            # for i in range(len(jobs_struct)):
            #     print(f"Job {i+1} : ")
            #     for j in range(len(jobs_struct[i])):
            #         print(f"O {j+1} : ", end="")
            #         for s in range(len(jobs_struct[i][j])):
            #             print(f"{jobs_struct[i][j][s]} ", end="")
            #         print()

        elif line.strip() == "<precedence constraints of operations>":

            line = file.readline().strip()
            while line != "": # fin de lecture de la section
                # print("line=", line)
                
                
                if line[0] == "J": # contrainte de précédence entre opérations d'un même jobs
                    # print("ici")
                    job_index = int(line.split("J")[1]) # permet de savoir quel job est considéré
                    # print("job_index=", job_index)
                    line = file.readline().strip()
                    while line != "" and line[0] != "J" and line[0] != "<":  # tant que contrainte d'operation
                        prec_constr = line.split(",")
                        # print("prec_constr=", prec_constr)
                        constraints_precedence_operations[job_index-1][int(prec_constr[0])-1][int(prec_constr[1])-1] = 1
                        line = file.readline().strip()


                if line[0] == "<": # contrainte de précédence entre sous opérations d'une operation
                    # print("la")
                    operation_index = int(line.split("<")[1].split(">")[0]) # permet de savoir quelle opération est considéré
                    # print("job_index=", job_index, "operation_index=", operation_index)
                    line = file.readline().strip()
                    while line != "" and line[0] != "J" and line[0] != "<": # tant que contrainte de sous operation
                        prec_constr_sub_op = line.split(",")
                        # print("prec_constr_sub_op=", prec_constr_sub_op)
                        constraints_precedence_sub_operations[job_index-1][operation_index-1][int(prec_constr_sub_op[0])-1][int(prec_constr_sub_op[1])-1] = 1
                        line = file.readline().strip()
            
            res["constraints_precedence_operations"] = constraints_precedence_operations
            res["constraints_precedence_sub_operations"] = constraints_precedence_sub_operations

            # print("constraints_precedence_operations=", constraints_precedence_operations.shape)
            # print(constraints_precedence_operations)
            # print("-------------------------------")
            # print("constraints_precedence_sub_operations=", constraints_precedence_sub_operations.shape)
            # print(constraints_precedence_sub_operations)
            # print("END")        
    

    file.close()
    
    # res : nb_jobs, nb_professions, nb_sub_operations_profession, nb_sub_operations, sub_operations_difficulties, sub_operations_times, max_nb_operations, max_nb_sub_operations, nb_workers, levels_workers, difficulty_jobs, jobs_struct, constraints_precedence_operations, constraints_precedence_sub_operations
    return res # dict with all the data of the instance

def read_solution_file(filename):
    """
    Lit un ficheir sol de Gurobi et retourne une liste de tuples pour la classe Solution.
    """

    res = []
    found_e = False

    f = open(filename, 'r')
    for line in f:
        if line[0] != "#": 
            line = line.strip().split(" ") 
            name = line[0]
            value = line[1]
            for i in range(len(value)):
                if value[i] == "e":
                    value = float(value[:i]) * (10 ** int(value[i+1:]))
                    found_e = True
                    break
            if not found_e:
                value = float(value)
            res.append((name, value))
    f.close()
    return res

def plot_cognitive_load_tutors(solution, instance, verbose=False):
    """
    Affiche la charge cognitive liée à l'apprentissage pour les tuteurs pour chaque métier après chaque run du PL
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """

    if verbose:
        print("cognitive_load_tutors=")
        print(solution.cognitive_load_tutors) # size (nb_workers, nb_professions)

    for k in range(instance.nb_workers):
        plt.plot(solution.cognitive_load_tutors[k, :], marker='o', label=f'w{k+1}')
    
    plt.title('Cognitive load related to learning for tutors for each profession')
    plt.xlabel('Profession Index')
    plt.ylabel('Cognitive Load')
    plt.xticks(range(instance.nb_professions))
    plt.legend()
    plt.grid()
    plt.show()

def plot_cognitive_load_collaboration(solution, instance, verbose=False):
    """
    Affiche la charge cognitive liée à la collaboration pour les travailleurs pour chaque métier après chaque run du PL
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """

    if verbose:
        print("cognitive_load_collaboration=")
        print(solution.cognitive_load_collaboration) # size (nb_workers, nb_professions)

    for k in range(instance.nb_workers):
        plt.plot(solution.cognitive_load_collaboration[k, :], marker='o', label=f'w{k+1}')
    
    plt.title('Cognitive load related to collaboration for each profession')
    plt.xlabel('Profession Index')
    plt.ylabel('Cognitive Load')
    plt.xticks(range(instance.nb_professions))
    plt.legend()
    plt.grid()
    plt.show()

def plot_cognitive_load_apprentis(solution, instance, verbose=False):
    """
    Affiche la charge cognitive liée à l'apprentissage pour les apprentis pour chaque métier après chaque run du PL
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """

    if verbose:
        print("cognitive_load_apprentis=")
        print(solution.cognitive_load_apprentis) # size (nb_workers, nb_professions)

    for k in range(instance.nb_workers):
        plt.plot(solution.cognitive_load_apprentis[k, :], marker='o', label=f'w{k+1}')
    
    plt.title('Cognitive load related to learning for apprentices for each profession')
    plt.xlabel('Profession Index')
    plt.ylabel('Cognitive Load')
    plt.xticks(range(instance.nb_professions))
    plt.legend()
    plt.grid()
    plt.show()

def plot_cognitive_load_total(solution, instance, verbose=False):
    """
    Affiche la charge cognitive totale pour les travailleurs pour chaque métier après chaque run du PL
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """

    if verbose:
        print("cognitive_load_total=")
        print(solution.cognitive_load_total) # size (nb_workers, nb_professions)

    for k in range(instance.nb_workers):
        plt.plot(solution.cognitive_load_total[k, :], marker='o', label=f'w{k+1}')
    
    plt.title('Total cognitive load for each profession')
    plt.xlabel('Profession Index')
    plt.ylabel('Cognitive Load')
    plt.xticks(range(instance.nb_professions))
    plt.legend()
    plt.grid()
    plt.show()

def plot_levels_workers(solution, instance, verbose=False):
    """
    Affiche les niveaux de compétences des travailleurs pour chaque sous-opération apres chaque run du PL
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """


    levels_workers = np.zeros((instance.nb_workers, 2, instance.nb_professions)) # 2 pour initial et final levels of workers for each metier
    levels_workers[:, 0, :] = instance.levels_workers[:, :] # initial levels of workers for each profession
    levels_workers[:, 1, :] = solution.l[:, :] # levels of workers for each profession after run of the PL (initially equal to initial levels)

    
    if verbose :
        print("levels_workers=")
        print(levels_workers) # size (nb_workers, 2, nb_professions) : levels_workers[k, 0, m] = initial level of worker k for profession m, levels_workers[k, 1, m] = final level of worker k for profession m after run of the PL
        
    if instance.nb_workers == 1:
        plt.plot(levels_workers[0, 0, :], marker='o', label='Initial levels')
        plt.plot(levels_workers[0, 1, :], marker='s', label='Final levels')
        plt.title('Levels of Worker 1 for each profession')
        plt.xlabel('Profession Index')
        plt.ylabel('Level of Worker')
        plt.xticks(range(instance.nb_professions))
        plt.legend()
        plt.grid()
        plt.show()
    else :
        fig, axs = plt.subplots(instance.nb_workers)
        for k in range(instance.nb_workers):
            axs[k].plot(levels_workers[k, 0, :], marker='o', label=f'w{k+1} initial level')
            axs[k].plot(levels_workers[k, 1, :], marker='s', label=f'w{k+1} final level')
            axs[k].set_title(f'Levels of Worker {k+1} for each profession')
            axs[k].set_xlabel('Profession Index')
            axs[k].set_ylabel('Level of Worker')
            axs[k].set_xticks(range(instance.nb_professions))
            axs[k].legend()
            axs[k].grid()
        plt.tight_layout()
        plt.show()
    

# Mettre au propre la fonction suivante pour ne pas avoir des constantes en durs et pour éviter redondances
def gantt_chart(solution, instance, color=0, render="html", save_path=None, verbose=False):
    """ 
    Affiche le diagramme de Gantt pour une solution donnée et une instance du problème.
    Par défaut la coloration est faite par sous-opération.

    Args:
        solution (Solution) : Une solution de l'instance 
        instance (Instance) : Une instance du problème
        color (int) :  0 -> coloration par sous-opération.
                       1 -> coloration par opération.
                       2 -> coloration par job.
                       3 -> coloration par mode (seul, apprentissage, collaboratif)
                       pour choisir la coloration du diagramme de Gantt.

                            
        Returns:    
            None : Affiche le diagramme de Gantt
    """
    x = np.arange(solution.C_max)
    y = [ [] for _ in range(instance.nb_workers) ] # y[k] = [(start_time, sub_operation, processing_time), ...] for each worker k


    dico_mode_to_str = {0: "alone", 1: "learning", 2: "collaboratively"}
    ##### filling the list of tasks for each worker with their start time and processing time
    for i in range(instance.nb_jobs):
        for j in range(len(instance.jobs_struct[i])):
            for s in range(len(instance.jobs_struct[i][j])):
                for k in range(instance.nb_workers):
                    if solution.x[i, j, s, k] == 1:
                        start_time = solution.d[i, j, s, k]
                        sub_op = (i, j, s) # sub operation s of operation j of job i

                        elementary_task = int(instance.jobs_struct[i][j][s])
                        processing_time = solution.f[i, j, s, k] - start_time
                        metier = int(instance.sub_op_to_m[elementary_task])
                        level_worker = instance.levels_workers[k][metier]
                        difficulty_task = instance.sub_operations_difficulties[elementary_task]
                        for z in range(3):
                            if solution.z_auxilary[i, j, s, z] == 1:
                                mode = dico_mode_to_str[z]
                                if mode == "learning":
                                    if solution.is_tutor[i, j, s, k] == 1:
                                        mode += " (tutor)"
                                    else:
                                        mode += " (apprentice)"
                        # processing_time = instance.sub_operations_times[sub_op_index][0] # [0] pour le momnent à modif si 2 workers
                        y[k].append((start_time, sub_op, processing_time, elementary_task, metier, mode, level_worker, difficulty_task))


    



    
    ##### sorting the tasks for each worker by their start time
    for k in range(instance.nb_workers):
        y[k].sort()
        if verbose:
            print(f"Worker w{k+1} sorted tasks: ", y[k] ," : (start_time, operation, processing_time, metier, elementary_task, mode, level_worker, difficulty_task)")

    print(y[1])
    ########################################################################
    ########################################################################
    ##
    ## On à une matrice (nb_jobs, max_nb_operations, max_nb_sub_operations) 
    ## ex: 
    ##     J1 : so_111, so_112, so_113
    ##          so_121, so_122, so_123
    ##          so_131, so_132, so_133
    ##
    ##     J2 : so_211, so_212, so_213, so_214
    ##          so_221, so_222, so_223, so_224
    ##          so_231, so_232, so_233, so_234
    ##          so_241, so_242, so_243, so_244
    ##
    ########################################################################
    ########################################################################


    ##### plotting the Gantt chart
    df = pd.DataFrame(columns=["Task", "Start", "Finish", "Sub_operation" ,"Operation", "Job", "mode", "Level_worker", "Difficulty_task"])
    for k in range(instance.nb_workers): # for each worker k
        for task in y[k]: # for each task of worker k
            start_time, (i, j, s), processing_time, elementary_task, metier, mode, level_worker, difficulty_task = task
            finish_time = start_time + processing_time
            df_tmp = pd.DataFrame({"Task": [f"w{k+1}"],
                                   "Start": [start_time],# + 1e-10000],
                                   "Finish": [finish_time],
                                   "Elementary task": [elementary_task],
                                   "Metier": [metier],
                                   "Sub_operation": [f"({i+1},{j+1},{s+1})"],
                                   "Operation": ["(" + str(i+1) + "," + str(j+1) + ")"],
                                   "Job": ["J"+str(i+1)],
                                   "mode": [mode],
                                   "Level_worker": [level_worker],
                                   "Difficulty_task": [difficulty_task]
                                   })
            df = pd.concat([df, df_tmp], ignore_index=True) # ignore_index=True for following the index of df only, not df_tmp
    if verbose:
        print(df)
    
    
    
    
    if color == 0:
        color_print = "Sub_operation"
    elif color == 1:
        color_print = "Operation"
    elif color == 2:
        color_print = "Job"
    elif color == 3:
        color_print = "mode"
    else : 
        print("Quelle coloration souhaitez-vous pour le diagramme de Gantt ? (0 pour Sub_operation, 1 pour Operation, 2 pour Job, 3 pour mode)")
        return

    if color == 0:
        colours = []
        for key in df["Sub_operation"].unique(): # if we want to see colors of sub-operations
            # print("key=", key)
            # print(type(key))
            colours.append(f"#{hashlib.md5(str(key).encode()).hexdigest()[:6]}")

        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, colors=colours, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                              title=f"Gantt Chart (makespan= {solution.C_max})")#, legend_title=color_print)
        fig.update_layout(legend_title_text="Sub-Operation(i,j,s)")

    else :
        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                          title=f"Gantt Chart (makespan= {solution.C_max})")
        
        if color == 1:
            fig.update_layout(legend_title_text="Operation(i,j)")
        
        elif color == 2:
            fig.update_layout(legend_title_text="Job(i)") 

        elif color == 3:
            fig.update_layout(legend_title_text="Mode of execution")

    
    fig.layout.xaxis.type = "linear" # for having numeric x-axis instead of date
    if render == "notebook":
        fig.show()
    elif render == "html":
        fig.write_html('../results/gantt_chart.html', auto_open=True) 
    if save_path is not None:
        fig.write_html(save_path)

    return df

            
def plot_precedence_graph(instance):
    """
    Affiche le graphe de précédence des opérations de chaque Job d'une instance donnée.

    Args:
        instance (Instance) : Une instance du problème
    """
    G = [] # liste des graphes de précédence des opérations de chaque Job
    
    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i])
        G.append(nx.DiGraph()) # Graphe du Job i
        G[i].add_nodes_from([f"O{j+1}" for j in range(nb_op_job_i)]) # noeuds/opérations du graphe du Job i
        
        for j in range(nb_op_job_i):
            for j_prime in range(nb_op_job_i):
                if instance.constraints_precedence_operations[i][j][j_prime] == 1: # si j est un prédécesseur de j_prime
                    G[i].add_edge(f"O{j+1}", f"O{j_prime+1}") # ajout de l'arc j --> j_prime

        pos = nx.spring_layout(G[i]) 
        plt.figure(figsize=(8, 6))
        nx.draw(G[i], pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
        plt.title(f"Graphe de précédence des opérations du Job J{i+1}")
        plt.show()

def plot_precedence_graph_sub_operations(instance):
    """
    Affiche le graphe de précédence des sous-opérations de chaque opération de chaque Job d'une instance donnée.

    Args:
        instance (Instance) : Une instance du problème
    """
    
    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i]) # nombre d'opérations du Job i
        G = [nx.DiGraph() for _ in range(nb_op_job_i)] # un graphe de précédence pour chaque opération du Job i
        
        for j in range(nb_op_job_i):
            nb_sub_op_job_i_j = len(instance.jobs_struct[i][j]) # 10
            G[j].add_nodes_from([f"{i+1}_{j+1}_{s+1}" for s in range(nb_sub_op_job_i_j)])
            
            for s in range(nb_sub_op_job_i_j):
                for s_prime in range(nb_sub_op_job_i_j): # A voir plus tard : il ne peut pas avoir de boucle entre des sub-op donc peut considérer que s_prime > s pour éviter les redondances
                    if instance.constraints_precedence_sub_operations[i][j][s][s_prime] == 1:
                        G[j].add_edge(f"{i+1}_{j+1}_{s+1}", f"{i+1}_{j+1}_{s_prime+1}")
            
            pos = nx.spring_layout(G[j], k=0.5, iterations=20)
            plt.figure(figsize=(8, 6))
            nx.draw(G[j], pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=10, font_weight='normal', arrowsize=20)
            plt.title(f"Graphe de précédence des sous-opérations de l'opération O{j+1} du Job J{i+1}")
            plt.show()
            
    
if __name__ == "__main__":
    file_path = "../data/data_temp.test"
    data = read_file(file_path)
    print("--------------------------------------")
    print("PRINT FINAL")
    for key in data:
        print(f"{key} = ")
        print(data[key])
        print("--------------------------------------")

