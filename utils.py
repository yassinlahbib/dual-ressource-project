import numpy as np
import plotly.figure_factory as ff
import pandas as pd
from time import time

def read_file(file_path):
    file = open(file_path, "r")

    levels_workers = []
    difficulty_jobs = []

    
    for line in file:

        if line.strip() == "<number of jobs>":
            nb_jobs = int(file.readline().strip())


        elif line.strip() == "<maximal number of operations>":
            max_nb_operations = int(file.readline().strip())


        elif line.strip() == "<maximal number of sub-operations>":
            max_nb_sub_operations = int(file.readline().strip())

            difficulty_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_sub_operations))
            processing_time_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_sub_operations))

            constraints_precedence_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_operations)) # consideration que le nombre d'operations par job est de même ordre de grandeur pour faire matrice d'adjacence des contraintes de precedences
            constraints_precedence_sub_operations = np.zeros((nb_jobs, max_nb_operations, max_nb_sub_operations, max_nb_sub_operations))


        elif line.strip() == "<number of workers>":
            nb_workers = int(file.readline().strip())


        elif line.strip() == "<levels of workers>":
            levels = file.readline().strip().split(" ")
            for i in range(nb_workers):
                levels_workers.append(int(levels[i]))
            print("levels_workers=", levels_workers)


        elif line.strip() == "<difficulty of jobs>":
            for _ in range(nb_jobs):
                difficulty_jobs.append(int(file.readline().strip()))

        
        elif line.strip() == "<(processing time,difficulty) of operations>":
            job_index = 0
            print("-----------------------------------")
            print("-----------------------------------")
            print("-----------------------------------")
            
            while job_index <= nb_jobs :
                print("job_index=", job_index)
                line = file.readline().strip()
                if line == "": # fin de lecture de la section
                    break
                if line[0] == "J":
                    job_index += 1
                    operation_index = 0
                else:
                    operation = line.split(" ")
                    for sub_op_index in range(len(operation)): # toutes les sous opérations d'une opération sont sur la même ligne
                        operation[sub_op_index] = operation[sub_op_index].split(",")
                        process_time_sub_operation = int(operation[sub_op_index][0])
                        difficulty_sub_operation = int(operation[sub_op_index][1])
                        print(process_time_sub_operation, difficulty_sub_operation ,"-> job_index=", job_index, "operation_index=", operation_index, "sub_op_index=", sub_op_index)
                        processing_time_operations[job_index-1][operation_index][sub_op_index] = process_time_sub_operation
                        difficulty_operations[job_index-1][operation_index][sub_op_index] = difficulty_sub_operation
                    operation_index += 1


        elif line.strip() == "<precedence constraints of operations>":
            print("processing_time_operations=")
            print(processing_time_operations)
            print("difficulty_operations=")
            print(difficulty_operations)

            line = file.readline().strip()
            while line != "": # fin de lecture de la section
                print("line=", line)
                
                
                if line[0] == "J": # contrainte de précédence entre opérations d'un même jobs
                    print("ici")
                    job_index = int(line.split("J")[1]) # permet de savoir quel job est considéré
                    print("job_index=", job_index)
                    line = file.readline().strip()
                    while line != "" and line[0] != "J" and line[0] != "<":  # tant que contrainte d'operation
                        prec_constr = line.split(",")
                        print("prec_constr=", prec_constr)
                        constraints_precedence_operations[job_index-1][int(prec_constr[0])-1][int(prec_constr[1])-1] = 1
                        line = file.readline().strip()


                if line[0] == "<": # contrainte de précédence entre sous opérations d'une operation
                    print("la")
                    operation_index = int(line.split("<")[1].split(">")[0]) # permet de savoir quelle opération est considéré
                    print("job_index=", job_index, "operation_index=", operation_index)
                    line = file.readline().strip()
                    while line != "" and line[0] != "J" and line[0] != "<": # tant que contrainte de sous operation
                        prec_constr_sub_op = line.split(",")
                        print("prec_constr_sub_op=", prec_constr_sub_op)
                        constraints_precedence_sub_operations[job_index-1][operation_index-1][int(prec_constr_sub_op[0])-1][int(prec_constr_sub_op[1])-1] = 1
                        line = file.readline().strip()

            print("constraints_precedence_operations=", constraints_precedence_operations.shape)
            print(constraints_precedence_operations)
            print("-------------------------------")
            print("constraints_precedence_sub_operations=", constraints_precedence_sub_operations.shape)
            print(constraints_precedence_sub_operations)
            print("END")        
    

    file.close()
    return nb_jobs, max_nb_operations, max_nb_sub_operations, nb_workers, levels_workers, difficulty_jobs, difficulty_operations, processing_time_operations, constraints_precedence_operations, constraints_precedence_sub_operations


import hashlib
def gantt_chart(solution, instance, color_print="Job"):
    """ 
    Affiche le diagramme de Gantt pour une solution donnée et une instance du problème 

    Args:
        solution (Solution) : Une solution de l'instance 
        instance (Instance) : Une instance du problème

        Returns:
            None : Affiche le diagramme de Gantt
    """
    x = np.arange(solution.C_max)
    y = [ [] for _ in range(instance.nb_workers) ] # y[k] = [(start_time, sub_operation, processing_time), ...] for each worker k

    ##### filling the list of tasks for each worker with their start time and processing time
    for i in range(instance.nb_jobs):
        for j in range(instance.max_nb_operations):
            for s in range(instance.max_nb_sub_operations):
                for k in range(instance.nb_workers):
                    if solution.x[i, j, s, k] == 1:
                        start_time = solution.d[i, j, s, k]
                        sub_op = (i, j, s) # sub operation s of operation j of job i
                        processing_time = instance.processing_time_operations[i][j][s]
                        y[k].append((start_time, sub_op, processing_time))


    
    ##### sorting the tasks for each worker by their start time
    for k in range(instance.nb_workers):
        y[k].sort()
        print(f"Worker w{k+1} sorted tasks: ", y[k] ," : (start_time, operation, processing_time)")

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
    df = pd.DataFrame(columns=["Task", "Start", "Finish", "sub_operation" ,"Operation", "Job"])
    for k in range(instance.nb_workers): # for each worker k
        for task in y[k]: # for each task of worker k
            start_time, (i, j, s), processing_time = task
            finish_time = start_time + processing_time
            df_tmp = pd.DataFrame({"Task": [f"w{k+1}"],
                                   "Start": [start_time + 1e-1],
                                   "Finish": [finish_time],
                                   "sub_operation": [f"({i+1},{j+1},{s+1})"],
                                   "Operation": ["(" + str(i+1) + "," + str(j+1) + ")"],
                                   "Job": ["J"+str(i+1)]
                                   })
            df = pd.concat([df, df_tmp], ignore_index=True) # ignore_index=True for following the index of df only, not df_tmp
    print(df)
    
    
    
    if color_print == "sub_operation":
        colours = []
        for key in df["sub_operation"].unique(): # if we want to see colors of sub-operations
            print("key=", key)
            print(type(key))
            colours.append(f"#{hashlib.md5(str(key).encode()).hexdigest()[:6]}")

        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, colors=colours, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                              title=f"Gantt Chart (makespan= {solution.C_max})")#, legend_title=color_print)
        fig.update_layout(legend_title_text="Sub-Operation(i,j,s)")

    else :
        fig = ff.create_gantt(df, group_tasks=True, index_col=color_print, show_colorbar=True, showgrid_x=True, showgrid_y=True,
                          title=f"Gantt Chart (makespan= {solution.C_max})")
        
        if color_print == "Operation":
            fig.update_layout(legend_title_text="Operation(i,j)")
        
        elif color_print == "Job":
            fig.update_layout(legend_title_text="Job(i)") 

        else :
            print("Quelle coloration souhaitez-vous pour le diagramme de Gantt ? (sub_operation, Operation, Job)")
            return

    
    fig.layout.xaxis.type = "linear" # for having numeric x-axis instead of date
    fig.write_html('gantt_chart.html', auto_open=True) 
            

if __name__ == "__main__":
    file_path = "data_2.test"
    data = read_file(file_path)
    print(data)
