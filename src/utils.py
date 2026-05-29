import numpy as np
import plotly.figure_factory as ff
import plotly.colors as pc
import pandas as pd
import hashlib # for generating colors for the Gantt chart based on sub-operation index

import networkx as nx
from networkx.algorithms import bipartite

import random
import matplotlib.pyplot as plt

def read_file(file_path):
    file = open(file_path, "r")

    nb_sub_operations = 0
    res = dict()
    
    for line in file:

        if line.strip() == "<number of jobs>":
            nb_jobs = int(file.readline().strip())
            res["nb_jobs"] = nb_jobs
            # print("nb_jobs=", nb_jobs)


        elif line.strip() == "<number of professions>":
            nb_professions = int(file.readline().strip())
            res["nb_professions"] = nb_professions
            # print("nb_professions=", nb_professions)


        elif line.strip() == "<professions detailed>":
            line = file.readline().strip().split(" ")
            assert nb_professions == len(line), "Le nombre de corps de métier doit être égale à la taille du vecteur professions detailed"
            nb_task_in_profession = np.zeros(nb_professions)

            for s in range(nb_professions):
                nb_task_in_profession[s] = int(line[s])
            res["nb_task_in_profession"] = nb_task_in_profession
            # print("nb_task_in_profession=", nb_task_in_profession)
            
            nb_tasks = int(np.sum(nb_task_in_profession))
            res["nb_tasks"] = nb_tasks # peut être déduit de nb_task_in_profession
            res["nb_professions"] = nb_professions # peut etre déduit de nb_task_in_profession
            # print("nb_tasks=", nb_tasks)


        elif line.strip() == "<tasks(difficulty and times)>":
            # print("here !!!!!!!!!!!!!!!")
            dict_task_to_m = dict()
            tasks_difficulties = []
            tasks_times = np.zeros((nb_tasks, 3)) # 3 columns for doing alone, with learning or collaboratively
            index_task = 0

            # print("AVANT WHILE line =", line)
            while line != '': # fin de lecture de la section
                line = file.readline().strip()
                if line == "": # fin de lecture de la section
                    break
                # print("line=", line)
                if line[0] != "m": # read the profession index 
                    # print("line[0]=", line[0])
                    line = line.split(" ")
                    tasks_difficulties.append(int(line[0])) # difficulty of task s
                    tasks_times[index_task][0] = float(line[1]) # processing time of task s if done alone
                    tasks_times[index_task][1] = float(line[2]) # processing time of task s if done with learning effect
                    tasks_times[index_task][2] = float(line[3]) # processing time of task s if done collaboratively
                    dict_task_to_m[index_task] = m
                    # print("lineFIN =", line)
                    index_task += 1
                if line[0] == "m":
                    m = int(line.split("m")[1]) -1
            res["tasks_difficulties"] = np.array(tasks_difficulties)
            res["tasks_times"] = tasks_times

            # print("--------------------------------")
            # print("tasks_difficulties=")
            # print(tasks_difficulties)
            # print("--------------------------------")
            # print("tasks_times=")
            # print(tasks_times)
            # print("--------------------------------")

            # print("dict_task_to_m=", dict_task_to_m)
            res["dict_task_to_m"] = dict_task_to_m


        elif line.strip() == "<maximal number of operations>":
            max_nb_operations = int(file.readline().strip())
            res["max_nb_operations"] = max_nb_operations # max opération par jobs
            constraints_precedence_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_operations)) # consideration que le nombre d'operations par job est de même ordre de grandeur pour faire matrice d'adjacence des contraintes de precedences


        elif line.strip() == "<number of workers>":
            nb_workers = int(file.readline().strip())
            res["nb_workers"] = nb_workers
        

        elif line.strip() == "<levels workers>":
            levels_workers = np.zeros((nb_workers, nb_professions))
            for i in range(nb_workers):
                line = file.readline().strip().split(" ")
                for j in range(nb_professions):
                    levels_workers[i][j] = float(line[j])

            res["levels_workers"] = levels_workers
            # print("levels_workers=", levels_workers) 
        
        elif line.strip() == "<forgetting workers>":
            forgetting_workers = np.zeros((nb_workers, nb_professions))
            for i in range(nb_workers):
                line = file.readline().strip().split(" ")
                for j in range(nb_professions):
                    forgetting_workers[i][j] = float(line[j])

            res["forgetting_workers"] = forgetting_workers
            # print("forgetting_workers=", forgetting_workers)


        elif line.strip() == "<difficulty of jobs>":
            difficulty_jobs = np.zeros(nb_jobs)
            line = file.readline().strip().split(" ")
            for i in range(nb_jobs):
                difficulty_jobs[i] = int(line[i])
            # print("difficulty_jobs=", difficulty_jobs)
            res["difficulty_jobs"] = difficulty_jobs

        elif line.strip() == "<resale price of jobs>":
            resale_price_jobs = np.zeros(nb_jobs)
            line = file.readline().strip().split(" ")
            for i in range(nb_jobs):
                resale_price_jobs[i] = float(line[i])
            # print("resale_price_jobs=", resale_price_jobs)
            res["resale_price_jobs"] = resale_price_jobs

        elif line.strip() == "<jobs>":
            jobs_struct = [] # jobs_struct[i] = [operation1, operation2, ...]
            job_index = 0
            
            while job_index <= nb_jobs :
                # print("job_index=", job_index)
                line = file.readline().strip()
                if line == "": # fin de lecture de la section
                    break
                if line[0] == "J":
                    job_index += 1
                else:
                    operation = line.split(" ") # 1 4 5
                    jobs_struct.append([]) # pour ajouter l'opération courante à la structure du job courant
                    for i in range(len(operation)):
                        jobs_struct[-1].append(int(operation[i])-1) # on stocke l'index de la la tache dans la structure du job courant
 
            res["jobs_struct"] = jobs_struct
            # print("jobs_struct=", jobs_struct)


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

            res["constraints_precedence_operations"] = constraints_precedence_operations
            # print("constraints_precedence_operations=", constraints_precedence_operations.shape)
            # print(constraints_precedence_operations)
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

def bars_cognitive_load_total(solution, instance, verbose=False):
    """
    Affiche la charge cognitive totale pour chaque travailleurs sous forme de bars
    
    Args:
    solution (Solution) : Une solution de l'instance 
    instance (Instance) : Une instance du problème

    Returns:
        None : Affiche le graphique
    """

    res = np.sum(solution.cognitive_load_total, axis=1) # charge cognitive totale pour chaque travailleur k

    if verbose:
        print("cognitive_load_total_per_worker=")
        print(res) # size (nb_workers,)

    
    plt.bar([f'w{k+1}' for k in range(instance.nb_workers)], res)
    plt.title('Total cognitive load for each worker')
    plt.xlabel('Worker')
    plt.ylabel('Total Cognitive Load')
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
        for k in range(instance.nb_workers):
            plt.plot(levels_workers[k, 0, :], marker='o', label=f'w{k+1} initial level')
            plt.plot(levels_workers[k, 1, :], marker='s', label=f'w{k+1} final level')
            plt.title(f'Levels of Worker {k+1} for each profession')
            plt.xlabel('Profession Index')
            plt.ylabel('Level of Worker')
            plt.xticks(range(instance.nb_professions))
            plt.legend()
            plt.grid()
            plt.show()
        # fig, axs = plt.subplots(instance.nb_workers)
        # for k in range(instance.nb_workers):
        #     axs[k].plot(levels_workers[k, 0, :], marker='o', label=f'w{k+1} initial level')
        #     axs[k].plot(levels_workers[k, 1, :], marker='s', label=f'w{k+1} final level')
        #     axs[k].set_title(f'Levels of Worker {k+1} for each profession')
        #     axs[k].set_xlabel('Profession Index')
        #     axs[k].set_ylabel('Level of Worker')
        #     axs[k].set_xticks(range(instance.nb_professions))
        #     axs[k].legend()
        #     axs[k].grid()
        # plt.tight_layout()
        # plt.show()
    

# Source - https://stackoverflow.com/a/47872260
# Posted by ImportanceOfBeingErnest, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-21, License - CC BY-SA 4.0

from  matplotlib.colors import LinearSegmentedColormap


def resume_levels_workers(solution, instance, verbose=False):

    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    cmap = plt.cm.RdYlGn

    res_tot = solution.l - instance.levels_workers # gain de niveau de chaque worker pour chaque profession
    res_worker = np.sum(res_tot, axis=1) # gain de niveau total de chaque worker pour tous les métiers
    res_profession = np.sum(res_tot, axis=0) # gain de niveau total pour chaque profession pour tous les workers
    if verbose:
        print("Gains de niveaux de chaque travailleur pour chaque profession :")
        print(res_tot)
        print("Gains de niveaux pour chaque travailleur :")
        print(res_worker)
        print("Gains de niveaux pour chaque profession :")
        print(res_profession)

    plt.imshow(res_tot, cmap=cmap, aspect='auto')

    for k in range(instance.nb_workers):
        for m in range(instance.nb_professions):
            if res_tot[k, m] > 0:
                sign = "+"
            else:
                sign = ""
            plt.text(m, k, f" {sign}{res_tot[k, m]:.1f}", ha='center', va='center', color='black')

    plt.xticks(range(instance.nb_professions), [f'm{m+1}' for m in range(instance.nb_professions)])
    plt.yticks(range(instance.nb_workers), [f'w{k+1}' for k in range(instance.nb_workers)])
    plt.colorbar(label='Gain de niveau')
    plt.title("Gains de niveaux de chaque travailleur pour chaque profession")
    plt.xlabel("Profession")
    plt.ylabel("Worker")
    plt.show()


def resume_forgetting_effect_workers(solution, instance, verbose=False):

    # on inverse signification du vert et du rouge pour le forgetting effect : rouge pour perte de niveau et vert pour pas de perte de niveau
    cmap = LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    cmap = plt.cm.RdYlGn_r

    res_tot = solution.forgetting - instance.forgetting # perte de niveau de chaque worker pour chaque profession
    res_worker = np.sum(res_tot, axis=1) # perte de niveau total de chaque worker pour tous les métiers
    res_profession = np.sum(res_tot, axis=0) # perte de niveau total pour chaque profession pour tous les workers
    if verbose:
        print("Perte de niveaux de chaque travailleur pour chaque profession : (Forgetting Effect)")
        print(res_tot)
        print("Perte de niveaux pour chaque travailleur :")
        print(res_worker)
        print("Perte de niveaux pour chaque profession :")
        print(res_profession)

    plt.imshow(res_tot, cmap=cmap, aspect='auto')

    for k in range(instance.nb_workers):
        for m in range(instance.nb_professions):
            if res_tot[k, m] > 0:
                sign = "+"
            else:
                sign = ""
            plt.text(m, k, f" {sign}{res_tot[k, m]:.1f}", ha='center', va='center', color='black')

    plt.xticks(range(instance.nb_professions), [f'm{m+1}' for m in range(instance.nb_professions)])
    plt.yticks(range(instance.nb_workers), [f'w{k+1}' for k in range(instance.nb_workers)])
    plt.colorbar(label='Perte de niveau')
    plt.title("Pertes de niveaux - Forgetting Effect")
    plt.xlabel("Profession")
    plt.ylabel("Worker")
    plt.show()


def scheduling_to_df(solution, instance):

    x = np.arange(solution.C_max)
    y = [ [] for _ in range(instance.nb_workers) ] # y[k] = [(start_time, sub_operation, processing_time), ...] for each worker k


    # Pour savoir si une tache finit après le makespan
    penalty_makespan = solution.penalty_makespan
    # print("penalty_makespan=", penalty_makespan)
    # print("borne_sup_makespan=", solution.borne_sup_makespan)

    dico_mode_to_str = {0: "alone", 1: "learning", 2: "collaboratively", 3: "alone without levels"} 
    ##### filling the list of tasks for each worker with their start time and processing time
    for i in range(instance.nb_jobs):
        for j in range(len(instance.jobs_struct[i])):
            for k in range(instance.nb_workers):
                if solution.x[i, j, k] == 1:
                    start_time = solution.d[i, j, k]
                    op = (i, j)
                    elementary_task = int(instance.jobs_struct[i][j]) # tache élémentaire qui constitue l'opération j du job i
                    processing_time = solution.f[i, j, k] - start_time
                    metier = int(instance.task_to_m[elementary_task])
                    level_worker = instance.levels_workers[k][metier]
                    difficulty_task = instance.tasks_difficulties[elementary_task]
                    task_in_job_done_before_limit = solution.job_done_before_limit[i]

                    if solution.borne_sup_makespan != -1  :
                        if  start_time + processing_time > 0.001 + solution.borne_sup_makespan:
                            penalty_makespan_current_task = start_time + processing_time - solution.borne_sup_makespan # pénalité pour la tâche courante qui dépasse le makespan
                            bool_penalty_makespan_current_task = "after"
                        else:
                            penalty_makespan_current_task = 0
                            bool_penalty_makespan_current_task = "before"
                    else:
                        penalty_makespan_current_task = 0
                        bool_penalty_makespan_current_task = "before"

                    for z in range(4): # 4 modes : seul, apprentissage, collaboratif, seul sans levels
                        if solution.z_auxilary[i, j, z] == 1:
                            mode = dico_mode_to_str[z]
                            if mode == "learning":
                                if solution.is_tutor[i, j, k] == 1:
                                    mode += " (tutor)"
                                else:
                                    mode += " (apprentice)"
                    # processing_time = instance.sub_operations_times[sub_op_index][0] # [0] pour le momnent à modif si 2 workers
                    y[k].append((start_time, op, processing_time, elementary_task, metier, mode, level_worker, difficulty_task, penalty_makespan_current_task, bool_penalty_makespan_current_task))

    
    ##### sorting the tasks for each worker by their start time
    for k in range(instance.nb_workers):
        y[k].sort()
        # if verbose:
        #     print(f"Worker w{k+1} sorted tasks: ", y[k] ," : (start_time, operation, processing_time, metier, elementary_task, mode, level_worker, difficulty_task)")


    ##### plotting the Gantt chart
    df = pd.DataFrame(columns=["Task", "Start", "Finish", "Finish (var. f)", "Processing time", "Operation", "Job", "mode", "Level_w", "Difficulty_task", "penalty_C_max", "bool_penalty_C_max", "Job_done_before_limit"]) # dataframe for the Gantt chart, with columns for task, start time, finish time, processing time, operation, job, mode of execution, level of worker and difficulty of task
    for k in range(instance.nb_workers): # for each worker k
        for task in y[k]: # for each task of worker k
            start_time, (i, j), processing_time, elementary_task, metier, mode, level_worker, difficulty_task, penalty_makespan_current_task, bool_penalty_makespan_current_task = task
            finish_time = start_time + processing_time
            df_tmp = pd.DataFrame({"Task": [f"w{k+1}"],
                                   "Start": [start_time],
                                   "Finish": [finish_time],
                                   "Finish (var. f)" : [solution.f[i, j, k]],
                                   "Elementary task": [elementary_task],
                                   "Metier": [metier],
                                   "Processing time": [processing_time],
                                   "Operation": ["(" + str(i+1) + "," + str(j+1) + ")"],
                                   "Job": ["J"+str(i+1)],
                                   "mode": [mode],
                                   "Level_w": [level_worker],
                                   "Difficulty_task": [difficulty_task],
                                   "penalty_C_max": [penalty_makespan_current_task],
                                   "bool_penalty_C_max": [bool_penalty_makespan_current_task],
                                   "Job_done_before_limit": ["before" if solution.job_done_before_limit[i] else "after"]
                                   })
            df = pd.concat([df, df_tmp], ignore_index=True) # ignore_index=True for following the index of df only, not df_tmp

    return df

# Mettre au propre la fonction suivante pour ne pas avoir des constantes en durs et pour éviter redondances
def gantt_chart(df, color=0, render="html", save_path=None, separate_little=False):
    """ 
    Affiche le diagramme de Gantt pour une solution donnée et une instance du problème.
    Par défaut la coloration est faite par sous-opération.

    Args:
        df (pd.DataFrame) : Un DataFrame contenant les données du diagramme de Gantt, avec les colonnes suivantes :
                            "Task", "Start", "Finish", "Finish (var. f)", "Processing time", "Operation", "Job", "mode", "Level_w", "Difficulty_task"
        color (int) :  0 -> coloration par tache.
                       1 -> coloration par opération.
                       2 -> coloration par job.
                       3 -> coloration par mode (seul, apprentissage, collaboratif)
                       pour choisir la coloration du diagramme de Gantt.

                            
        Returns:    
            None : Affiche le diagramme de Gantt
    """

    if df.empty:
        print("No tasks to display in the Gantt chart.")
        return df
    

    c_max = df["Finish"].max() if not df.empty else 0
    if separate_little:
        df["Start"] = df["Start"] + 0.01    
    
    if color == 0:
        color_print = "Operation"
    elif color == 1:
        color_print = "bool_penalty_C_max" # Par opération si elle est éxécutée avant ou après le time limite accordé
    elif color == 2:
        color_print = "Job"
    elif color == 3:
        color_print = "mode"
    elif color == 4:
        color_print = "Job_done_before_limit" # Si l'opération aapartient à un job finit avant ou après le time limite accordé
    else : 
        print("Quelle coloration souhaitez-vous pour le diagramme de Gantt ? (0 pour Sub_operation, 1 pour Operation, 2 pour Job, 3 pour mode)")
        return

    if color == 0:
        unique_ops = sorted(df["Operation"].unique())

        palette = pc.qualitative.Plotly + pc.qualitative.D3 + pc.qualitative.Set3
        color_map = {op: palette[i % len(palette)] for i, op in enumerate(unique_ops)}
        print("color_map=", color_map)
        # colours = []
        # for key in df["Elementary task"].unique(): # if we want to see colors of tasks
        #     # print("key=", key)
        #     # print(type(key))
        #     colours.append(f"#{hashlib.md5(str(key).encode()).hexdigest()[:6]}")

        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, colors=color_map, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                              title=f"Gantt Chart (makespan= {c_max})")#, legend_title=color_print)
        fig.update_layout(legend_title_text="Operation(i,j)")

    else :
        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                          title=f"Gantt Chart (makespan= {c_max})")
        
        if color == 1:
            fig.update_layout(legend_title_text="feasible task")

        elif color == 2:
            fig.update_layout(legend_title_text="Job(i)") 

        elif color == 3:
            fig.update_layout(legend_title_text="Mode of execution")

        elif color == 4:
            fig.update_layout(legend_title_text="Job done before limit")

    
    fig.layout.xaxis.type = "linear" # for having numeric x-axis instead of date
    if render == "notebook":
        fig.show("png")

    if render == "interactif":
        fig.show()
    elif render == "html":
        fig.write_html('../results/gantt_chart.html', auto_open=True) 
    if save_path is not None:
        fig.write_html(save_path)

    # return df
            
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
            

# pas suffisant doit avoir plus d'assertion comme le fait que la tache en collab est bien fait a deux etc..  
def check_df(df):
    print(" ========== CHECKING CONDITIONS FOR GANTT CHART ========== ")
    for i in range(len(df)):
        if df["mode"][i] == "alone" and  df["Level_w"][i] < df["Difficulty_task"][i]:
            print("Condition 1 not satisfied for task ", i)
            print("not satisfied condition : mode alone and level worker < difficulty task (line :", i, ")")
            return False

        if df["mode"][i] == "learning (tutor)" and df["Level_w"][i] < df["Difficulty_task"][i]:
            print("Condition 2 not satisfied for task ", i)
            print("not satisfied condition : mode learning (tutor) and level worker < difficulty task (line :", i, ")")
            return False

        if df["mode"][i] == "learning (apprentice)" and df["Level_w"][i] > df["Difficulty_task"][i]:
            print("Condition 3 not satisfied for task ", i)
            print("not satisfied condition : mode learning (apprentice) and level worker > difficulty task (line :", i, ")")
            return False

        if df["mode"][i] == "collaboratively" and df["Level_w"][i] < df["Difficulty_task"][i]:
            print("Condition 4 not satisfied for task ", i)
            print("not satisfied condition : mode collaboratively and level worker < difficulty task (line :", i, ")")
            return False

        if df["mode"][i] == "alone without levels" and df["Level_w"][i] >= df["Difficulty_task"][i]:
            print("Condition 5 not satisfied for task ", i)
            print("not satisfied condition : mode alone without levels and level worker >= difficulty task (line :", i, ")")
            return False

    print("All conditions are satisfied for the Gantt chart.")
    return True




# CHECK PARETO DOMINANCE (ALL OBJECTIVES ARE MAXIMIZED)

def get_pareto_optimal(x):
    """
    Retourne les points de x qui sont Pareto optimaux.
    Arg :
        x : np.ndarray of shape (n_solutions, 3)
            x[0] = profit (maximize)
            x[1] = skills (maximize)
            x[2] = - cognitive load (maximize)
    """
    
    idx_dominated = []

    # point dans l'espace des critères
    if len(x.shape) == 2 : # size (n_solutions, 3)

        for i in range(len(x)):
            if i not in idx_dominated:

                for j in range(i+1, len(x)):
                    if j not in idx_dominated:

                        if domine(x[i], x[j]):
                            idx_dominated.append(j)
                            # print("Point ", j, " is dominated by point ", i)
                        elif domine(x[j], x[i]):
                            idx_dominated.append(i)
                            # print("Point ", i, " is dominated by point ", j)
                        else:
                            continue
        
        # print("idx_dominated=", idx_dominated)
        idx_dominated = list(set(idx_dominated)) # pour éviter les redondances
        for i in range(len(idx_dominated)-1, -1, -1): # from len -1 to 0
            # print("Point ", idx_dominated[i], " is dominated by another point.")
            x = np.delete(x, idx_dominated[i], axis=0)
        return x, idx_dominated

    # point dans l'espace des critères et dans l'espace de lorenz
    if len(x.shape) == 3 : # size (n_solutions, 2, nb_objectives)

        for i in range(len(x)):
            if i not in idx_dominated:

                for j in range(i+1, len(x)):
                    if j not in idx_dominated:

                        if domine(x[i,1], x[j,1]):
                            idx_dominated.append(j)
                        elif domine(x[j,1], x[i,1]):
                            idx_dominated.append(i)
                        else:
                            continue
        
        # print("idx_dominated=", idx_dominated)
        for i in range(len(idx_dominated)-1, -1, -1): # from len -1 to 0
            # print("Point ", idx_dominated[i], " is dominated by another point.")
            x = np.delete(x, idx_dominated[i], axis=0)
        return x, idx_dominated

def domine(x, y):
    """
    Retourne True si x domine y, False sinon.
    Arg :
        x : np.ndarray of shape (3,)
            x[0] = profit (maximize)
            x[1] = skills (maximize)
            x[2] = - cognitive load (maximize)
        y : np.ndarray of shape (3,)
            y[0] = profit (maximize)
            y[1] = skills (maximize)
            y[2] = - cognitive load (maximize)
    """

    return (x[0] >= y[0] and x[1] >= y[1] and x[2] >= y[2]) and (x[0] > y[0] or x[1] > y[1] or x[2] > y[2])

def check_dominance_pareto(x):
    """
    Retourne True si aucun point de x ne domine un autre point de x, False sinon.
    Arg :
        x : np.ndarray of shape (n_solutions, 3)
            x[0] = profit (maximize)
            x[1] = skills (maximize)
            x[2] = - cognitive load (maximize)
    """

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if domine(x[i], x[j]) or domine(x[j], x[i]):
                print("Points ", i, " and ", j, " are not Pareto optimal.")
                return False
    return True

def get_lorenz_vector(x):
    """
    Retourne le vecteur de Lorenz de x.
    Arg :
        x : np.ndarray of shape (n_solutions, 3)
            x[0] = profit (maximize)
            x[1] = skills (maximize)
            x[2] = - cognitive load (maximize)
    """

    lorenz_vector = []
    for i in range(len(x)):
        sorted_x = np.sort(x[i])
        lorenz_vector.append((x[i], np.cumsum(sorted_x)))
    lorenz_vector = np.array(lorenz_vector)
    return lorenz_vector # size (n_solutions, 2, nb_objectives)


# VIZUALIZATION OF EACH OPERATION, EACH WORKER CAN OPERATE
def visualization_before_scheduling(instance, not_qualified=False):
    

    res = instance.qualified_workers_for_task(verbose=False)
    # print("res=", type(res))
    # print("res=", len(res))
    qualified_workers, not_qualified_but_at_most_one = res[0], res[1]
    # print("qualified_workers=", qualified_workers)
    # print("not_qualified_but_at_most_one=", not_qualified_but_at_most_one)
    # for i in range(instance.nb_jobs):
    #     for j in range(len(instance.jobs_struct[i])):
    #         if len(qualified_workers[i][j]) == 0 :
    #             print("pas de worker qualifié pour faire l'opération ", (i+1, j+1))
    #             if len(not_qualified_but_at_most_one[i][j]) > 0:
    #                 print("mais il y a des workers qui ont au plus un niveau de différence pour faire l'opération ", (i+1, j+1), " : ", not_qualified_but_at_most_one[i][j])



    G_all_jobs = nx.DiGraph() # graphe de précédence de toutes les opérations de tous les jobs
    
    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i])
        G_all_jobs.add_nodes_from([(f"{i+1},{j+1}", {"workers": qualified_workers[i][j]}) for j in range(nb_op_job_i)])
        for j in range(nb_op_job_i):
            for j_prime in range(nb_op_job_i):
                if instance.constraints_precedence_operations[i][j][j_prime] == 1: # si j est un prédécesseur de j_prime
                    G_all_jobs.add_edge(f"{i+1},{j+1}", f"{i+1},{j_prime+1}") # ajout de l'arc j --> j_prime

    # color des noeuds en fonction de si des workers sont qualifiés pour faire l'opération
    for i in range(instance.nb_jobs):
        for j in range(len(instance.jobs_struct[i])):
            if len(qualified_workers[i][j]) > 0 :
                G_all_jobs.nodes[f"{i+1},{j+1}"]['node_color'] = 'green'
            else : 
                G_all_jobs.nodes[f"{i+1},{j+1}"]['node_color'] = 'red'

                if not_qualified:
                    G_all_jobs.nodes[f"{i+1},{j+1}"]['at_most_diff_of_one'] = not_qualified_but_at_most_one[i][j]
                    if len(not_qualified_but_at_most_one[i][j]) > 0:
                        G_all_jobs.nodes[f"{i+1},{j+1}"]['node_color'] = 'orange'
                    else :
                        G_all_jobs.nodes[f"{i+1},{j+1}"]['node_color'] = 'red'

                    

    ColorLegend = {
        "job with qualified workers": 'green',
        "job without qualified workers": 'red'

    }

    if not_qualified:
        ColorLegend["job have at most one level of difference"] = 'orange'

    
    # for n in G_all_jobs.nodes():
    #     print(n, G_all_jobs.nodes[n])


    legend_elements = []
    print("ColorLegend=", ColorLegend)
    for i in range(len(ColorLegend)):
        label = list(ColorLegend.keys())[i]
        color = list(ColorLegend.values())[i]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))
                


    color_map = [G_all_jobs.nodes[node]['node_color'] for node in G_all_jobs.nodes()]
    
    # pos_all_jobs = nx.spring_layout(G_all_jobs)
    # ======= POSITIONS OF NODES AND LABELS ======
    pos_all_jobs = {}
    x_spacing = 2.0
    y_spacing = 1.5

    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i])

        #ligne job i
        y = -i * y_spacing

        for j in range(nb_op_job_i):
            node = f"{i+1},{j+1}"

            # colonne operation j 
            x = j * x_spacing

            pos_all_jobs[node] = (x, y)
    
    pos_all_jobs = {
        node: (x, y-0.25) for node, (x, y) in pos_all_jobs.items()
    }

    plt.figure(figsize=(16, 8))

    # ====== PLOT RESALE PRICE OF JOBS ======
    for i in range(instance.nb_jobs):
        x = pos_all_jobs[f"{i+1},1"][0] # x de la première opération de chaque job
        y = pos_all_jobs[f"{i+1},1"][1] # y de la première opération de chaque job
        
        plt.text(x, y+0.3, 
                 f"{instance.resale_price_jobs[i]:.1f} EUR",
                   fontsize=10, 
                   fontweight='bold', 
                   color='blue'
                )
        




    # plt.subplots_adjust(right=1) # pour laisser de la place à la légende à droite du graphique

    labels = {node: f"{node}\n{G_all_jobs.nodes[node]['workers']}" for node in G_all_jobs.nodes() if len(G_all_jobs.nodes[node]['workers']) > 0}
    labels.update({node: f"{node}\n\n{G_all_jobs.nodes[node]['at_most_diff_of_one']}" for node in G_all_jobs.nodes() if len(G_all_jobs.nodes[node].get('at_most_diff_of_one', [])) > 0})

    nx.draw(G_all_jobs, pos_all_jobs, with_labels=True, labels=labels, node_color=color_map)
    
    plt.legend(handles=legend_elements,
                loc ="upper right",
                title="Qualification of workers for operations",
                bbox_to_anchor=(1.02, 0.5)
            )
    

    plt.title(f"Graphe de précédence de toutes les opérations de tous les jobs")
    plt.show()

def visualization_after_scheduling(instance, solution, df):

    # to know if a job is done or not,
    jobe_done = solution.job_done # size (nb_jobs,) : job_done[i] = 1 if job i is done, 0 otherwise
    

    # ====== GRAPH CREATION WITH LABEL JOB DONE ======
    G_all_jobs = nx.DiGraph() # graphe de précédence de toutes les opérations de tous les jobs
    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i])
        G_all_jobs.add_nodes_from([(f"{i+1},{j+1}", {"job_done": jobe_done[i]}) for j in range(nb_op_job_i)])
        for j in range(nb_op_job_i):
            for j_prime in range(nb_op_job_i):
                if instance.constraints_precedence_operations[i][j][j_prime] == 1: # si j est un prédécesseur de j_prime
                    G_all_jobs.add_edge(f"{i+1},{j+1}", f"{i+1},{j_prime+1}") # ajout de l'arc j --> j_prime


    ColorLegend = {
        "alone": 'dodgerblue',
        "learning": 'magenta',
        "collaboratively": 'green',
        "alone without levels": 'orange',
        "job not done": 'red'
    }

    # groupement par opération pour faciliter l'iteration sur le df
    df_grouped = (
        df[["Operation", "mode", "Task", "bool_penalty_C_max"]]
        .groupby(["Operation"])
        .agg({
            "mode": "first", # prendre le premier mode d'execution pour chaque operation (identique une opération)
            "Task": list, #liste des workers associés
            "bool_penalty_C_max": "first" # prendre la première valeur de la colonne bool_penalty_C_max pour chaque operation
        })
    ).reset_index()

    for i in range(len(df_grouped)):
        op = df_grouped.loc[i, "Operation"]
        mode = df_grouped.loc[i, "mode"]
        w_list = df_grouped.loc[i, "Task"] # liste des workers qui ont effectué l'opération (i,j)
        bool_penalty_C_max = df_grouped.loc[i, "bool_penalty_C_max"]
        if mode == "alone":
            color = ColorLegend["alone"]
        elif mode.startswith("learning"):
            color = ColorLegend["learning"]
        elif mode == "collaboratively":
            color = ColorLegend["collaboratively"]
        elif mode == "alone without levels":
            color = ColorLegend["alone without levels"]
        else:
            color = 'white' # default color if mode is not recognized

        # attribution des labels et des couleurs de chaque noeuds
        i, j = op.strip("()").split(",")
        G_all_jobs.nodes[f"{i},{j}"]['node_color'] = color
        G_all_jobs.nodes[f"{i},{j}"]['workers'] = w_list
        if bool_penalty_C_max == "after":
            G_all_jobs.nodes[f"{i},{j}"]["limit_makespan_exceeded"] = True
        else:
            G_all_jobs.nodes[f"{i},{j}"]["limit_makespan_exceeded"] = False

    # ======= POSITIONS OF NODES AND LABELS ======
    pos_all_jobs = {}
    x_spacing = 2.0
    y_spacing = 1.5

    for i in range(instance.nb_jobs):
        nb_op_job_i = len(instance.jobs_struct[i])

        #ligne job i
        y = -i * y_spacing

        for j in range(nb_op_job_i):
            node = f"{i+1},{j+1}"

            # colonne operation j 
            x = j * x_spacing

            pos_all_jobs[node] = (x, y)
    
    pos_labels = {
        node: (x, y-0.25) for node, (x, y) in pos_all_jobs.items()
    }

        
    #====== COLOR FOR NODES OR JOB NOT DONE ======
    for i in range(instance.nb_jobs):
        if jobe_done[i] == 0:
            for j in range(len(instance.jobs_struct[i])):
                G_all_jobs.nodes[f"{i+1},{j+1}"]['node_color'] = 'red'

    
    #===== LEGEND FOR COLORS NODES ====== 
    legend_elements = []
    for i in range(len(ColorLegend)):
        label = list(ColorLegend.keys())[i]
        color = list(ColorLegend.values())[i]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))
        
   #===== LEGEND FOR NODES WITH TIME NO FINISH AT TIME  ====== 
    legend_elements.append(
        plt.Line2D(
            [0], [0],
            marker='o', 
            color='darkred', 
            label="task not finished at time limit", 
            markerfacecolor='white', # pour avoir que le contoure coloré en rouge et pas le remplissage
            markersize=10, 
            markeredgewidth=3 # pour avoir un ring plus épais et visible
        )
    )

    #====== COLOR MAP FOR NODES ======
    color_map = [G_all_jobs.nodes[node]['node_color'] for node in G_all_jobs.nodes()]


    # ====== LABELS OF WORKERS FOR EACH NODE ======
    worker_labels = {}
    for n, d in G_all_jobs.nodes(data=True):
        if "workers" in d :
            worker_labels[n] = [str(n)] + d['workers'] + ["*"] if d.get("limit_makespan_exceeded", False) else [str(n)] +d['workers'] # ajouter une indication si la tâche est responsable du dépassement du makespan
        else:
            worker_labels[n] = [str(n)] # Pour les jobs not done pas de workers associés


    labels = {node: f"{node}\n{G_all_jobs.nodes[node].get('workers', '')}" for node in G_all_jobs.nodes()}



    plt.figure(figsize=(12, 8))

    # ====== PLOT RESALE PRICE OF JOBS ======
    for i in range(instance.nb_jobs):
        x = pos_all_jobs[f"{i+1},1"][0] # x de la première opération de chaque job
        y = pos_all_jobs[f"{i+1},1"][1] # y de la première opération de chaque job
        if jobe_done[i] == 0:
            color_price = 'red'
        else:
            color_price = 'darkgreen'
            for j in range(len(instance.jobs_struct[i])):
                if G_all_jobs.nodes[f"{i+1},{j+1}"].get("limit_makespan_exceeded", False):
                    color_price = 'peru'
                    break

        plt.text(x, y+0.3, 
                 f"{instance.resale_price_jobs[i]:.1f} EUR",
                   fontsize=10, 
                   fontweight='extra bold', 
                   color=color_price
                )

    node_size = 300
    ring_size = 3*node_size # + 200 

    edges = nx.draw_networkx_edges(
        G_all_jobs, 
        pos_all_jobs, 
        edge_color='black', 
        width=1.5
    )

    nodes = nx.draw_networkx_nodes(G_all_jobs,
            pos_all_jobs,
            node_color=color_map,
            node_size=node_size
    )


    #seuelemtn les noeud avec label limit_makespan_exceeded = True ont un ring gris pour indiquer qu'ils sont responsables du dépassement du makespan, les autres noeuds n'ont pas de ring
    G_all_jobs_rings = nx.Graph()
    for n, d in G_all_jobs.nodes(data=True):
        if d.get("limit_makespan_exceeded", False):
            G_all_jobs_rings.add_node(n, **d) # ajouter les mêmes attributs que dans G_all_jobs pour pouvoir les utiliser pour les labels

    rings = nx.draw_networkx_nodes(
        G_all_jobs_rings,
        pos_all_jobs,
        node_size=ring_size,
        node_color='darkred'
    )

    nx.draw_networkx_labels(G_all_jobs,
                            pos_labels,
                            labels=labels,
                            font_color='black'
    )

    plt.legend(handles=legend_elements,
                loc ="center left", 
                title="Mode of execution and job status", 
                bbox_to_anchor=(1.02, 0.5))
    
    nodes.set_zorder(2)
    rings.set_zorder(1)

    plt.title(f"Graphe de précédence de toutes les opérations de tous les jobs")
    plt.show()



def visualization_list_of_solutions(instance, solution_list, doing=True):
    """
    doing = True : visualiser les tâches faites par chaque solution
    doing = False : visualiser les tâches non faites par chaque solution
    """
    
    G = nx.Graph()
    nb_jobs = instance.nb_jobs

    for i in range(len(solution_list)):
        s_curr = solution_list[i]
        G.add_nodes_from([(f"s{i}", {"objectifs": list(s_curr.objective_values.values()) })], bipartite=0)
    
    for i in range(nb_jobs):
        G.add_nodes_from([(f"J{i}", {"diff_job": instance.difficulty_jobs[i]})], bipartite=1)


    for node in G.nodes(data=True):
        # print("node=", node)
        # print(node[0])
        # print(node[1])
        if node[1]["bipartite"] == 0:
            G.nodes[node[0]]["subset"] = 0
        else:
            G.nodes[node[0]]["subset"] = 1

    colorMAP = []
    # couleur basé sur indices de la solution
    for i in range(len(solution_list)):
        colorMAP.append(plt.cm.tab10(i % 10)) 

    for i in range(len(solution_list)):
        s_curr = solution_list[i]
        for j in range(nb_jobs):
            if doing == True :
                if s_curr.job_done[j] == 1:
                    G.add_edge(f"s{i}", f"J{j}")
                    G.edges[f"s{i}", f"J{j}"]['color'] = colorMAP[i]
            else :
                if s_curr.job_done[j] == 0:
                    G.add_edge(f"s{i}", f"J{j}")
                    G.edges[f"s{i}", f"J{j}"]['color'] = colorMAP[i]



    plt.figure(figsize=(12, 8))
    pos = nx.multipartite_layout(G)
    nx.draw(G,
            pos,
            with_labels=True,
            edge_color=[G.edges[edge]['color'] for edge in G.edges()]
            )


    



if __name__ == "__main__":
    file_path = "../data/data_temp.test"
    data = read_file(file_path)
    print("--------------------------------------")
    print("PRINT FINAL")
    for key in data:
        print(f"{key} = ")
        print(data[key])
        print("--------------------------------------")

