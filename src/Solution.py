import numpy as np
import pandas as pd
from Instance import *
from Constante import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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
        self.Level_min = np.zeros((instance.nb_jobs, instance.max_nb_operations)) # Level_min[i,j] = niveau du worker le plus faible sur l'opertion ij
        self.Delta_min = np.zeros((instance.nb_jobs, instance.max_nb_operations, instance.nb_workers)) # Delta_min[i,j,k] = Permet de'identifier le worker le plus faible sur l'opération ij parmi les workers qui font l'opération ij
        self.task_done = np.zeros((instance.nb_jobs, instance.max_nb_operations)) # task_done[i,j] = 1 si l'opération j du job i est faite, 0 sinon


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

            elif v[0][:9] == "job_done[":
                indices = v[0][9:-1].split(",")
                i = int(indices[0])
                self.job_done[i] = v[1]

            elif v[0][:2] == "y[":
                indices = v[0][2:-1].split(",")
                i, k = int(indices[0]), int(indices[1])
                self.y[i, k] = v[1]
            
            elif v[0] == "BORNE_SUP_MAKESPAN":
                print("Borne sup makespan:", v[1])
                self.borne_sup_makespan = v[1]

            elif v[0][:10] == "Level_min[":
                indices = v[0][10:-1].split(",")
                i, j = int(indices[0]), int(indices[1])
                self.Level_min[i, j] = v[1]

            elif v[0][:10] == "Delta_min[":
                indices = v[0][10:-1].split(",")
                i, j, k = int(indices[0]), int(indices[1]), int(indices[2])
                self.Delta_min[i, j, k] = v[1]

            elif v[0][:10] == "task_done[":
                indices = v[0][10:-1].split(",")
                i, j = int(indices[0]), int(indices[1])
                self.task_done[i, j] = v[1]

            # elif v[0][:21] == "penalty_makespan_job[":
            #     indices = v[0][21:-1].split(",")
            #     i = int(indices[0])
            #     self.penalty_makespan_job[i] = v[1]

            # elif v[0][:21] == "job_done_before_limit":
            #     indices = v[0][22:-1].split(",")
            #     i = int(indices[0])
            #     self.job_done_before_limit[i] = v[1]


            # elif v[0] == "penalty_makespan":
            #     indices = v[0][17:-1].split(",")
            #     self.penalty_makespan = v[1]

            # elif v[0][:16] == "penalty_deadline":
            #     indices = v[0][17:-1].split(",")
            #     i, j = int(indices[0]), int(indices[1])
            #     self.penalty_deadline[i, j] = v[1]

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

    def get_nb_jobs_done(self):
        return int(np.sum(self.job_done))

    def get_nb_jobs_beginned_but_not_done(self):
        jobs_beginned = self.task_done.sum(axis=1) > 0
        return int(np.sum(jobs_beginned & (self.job_done == 0)))



    def which_jobs_are_completed(self, verbose=False):
        print(" ========== Checking which jobs are completed: ==========")
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

    def _resale_price_job_done_plus_task_done(self, ponderation_task_done="one"):
        """
        Calcule le prix de revente des jobs terminés, pondéré par la difficulté des tâches si spécifié.
        Args:
            ponderation_task_done: "one" pour pondérer chaque tâche terminée par 1,
                                   "difficulty" pour pondérer par la difficulté de la tâche.
        Returns:
            Le prix de revente total des jobs terminés.
        """
        benefit = 0
        for i in range(len(self.instance.jobs_struct)):
            benefit += self.instance.resale_price_jobs[i] * self.job_done[i]
            for j in range(len(self.instance.jobs_struct[i])):
                if ponderation_task_done == "one":
                    benefit += self.task_done[i, j]
                elif ponderation_task_done == "difficulty":
                    benefit += self.task_done[i, j] * self.instance.get_difficulty_and_metier_of_task(i, j)[0]
        return benefit

    def get_resale_price_job_done(self):
        """ Prix de revente des jobs terminés. """
        return np.sum(self.job_done * self.instance.resale_price_jobs)

    def get_nb_tasks_done(self):
        """ Nombre de taches terminées """
        return np.sum(self.task_done)
        
    def get_sum_cognitive_load_total(self):
        return np.sum(self.cognitive_load_total)
    
    def get_sum_skill_levels_rate(self):
        return np.sum(self.l) - np.sum(self.instance.levels_workers)

    def get_nb_tasks_done_solo(self):
        """ Nombre de taches terminées en solo """
        return np.sum(self.z_auxilary[:,:,0])
    
    def get_nb_tasks_done_teaching(self):
        """ Nombre de taches terminées en teaching """
        return np.sum(self.z_auxilary[:,:,1])
    
    def get_nb_tasks_done_collab(self):
        """ Nombre de taches terminées en collaboration """
        return np.sum(self.z_auxilary[:,:,2])

    def get_nb_tasks_done_solo_no_level(self):
        """ Nombre de taches terminées en solo sans le niveau requis """
        return np.sum(self.z_auxilary[:,:,3])

    def get_nb_jobs_done_with_some_task_done_solo_no_level(self):
        """ Nombre de jobs terminés avec au moins une tache faite en solo sans le niveau requis et la liste des jobs concernés """
        jobs = []
        for i in range(self.instance.nb_jobs):
            if self.job_done[i] == 1 and np.sum(self.z_auxilary[i,:,3]) > 0:
                jobs.append(i)
        return len(jobs), jobs

    def get_nb_jobs_with_some_task_done_solo_no_level(self):
        """ Nombre de jobs avec au moins une tache faite en solo sans le niveau requis et la liste des jobs concernés """
        jobs = []
        for i in range(self.instance.nb_jobs):
            if np.sum(self.z_auxilary[i,:,3]) > 0:
                jobs.append(i)
        return len(jobs), jobs

    def get_cognitive_load_of_workers_highest(self):
        """ Retourne le worker avec la charge cognitive la plus élevée et sa charge cognitive """
        max_load = np.max(self.cognitive_load_total)
        worker_index = np.argmax(np.sum(self.cognitive_load_total, axis=1))
        return worker_index, max_load

    def get_cognitive_load_tutors(self):
        """ Retourne la charge cognitive des tuteurs """
        return np.sum(self.cognitive_load_total, axis=1)

    def get_cognitive_load_apprentis(self):
        """ Retourne la charge cognitive des apprentis """
        return np.sum(self.cognitive_load_total, axis=1)

    def get_cognitive_load_collaboration(self):
        """ Retourne la charge cognitive de la collaboration """
        return np.sum(self.cognitive_load_total, axis=1)

    def get_worker_do_maximum_tasks(self):
        """ Retourne le worker qui a fait le plus de taches et le nombre de taches faites """
        max_tasks = np.max(np.sum(self.x, axis=(0, 1)))
        worker_index = np.argmax(np.sum(self.x, axis=(0, 1)))
        return worker_index, max_tasks
    
    def get_worker_do_maximum_jobs(self):
        """ Retourne le worker qui a fait le plus de jobs et le nombre de jobs faits """
        max_jobs = np.max(np.sum(self.job_done[:, np.newaxis] * self.x, axis=(0, 1)))
        worker_index = np.argmax(np.sum(self.job_done[:, np.newaxis] * self.x, axis=(0, 1)))
        return worker_index, max_jobs
    
    def get_worker_do_minimum_tasks(self):
        """ Retourne le worker qui a fait le moins de taches et le nombre de taches faites """
        min_tasks = np.min(np.sum(self.x, axis=(0, 1)))
        worker_index = np.argmin(np.sum(self.x, axis=(0, 1)))
        return worker_index, min_tasks

    def get_worker_do_minimum_jobs(self):
        """ Retourne le worker qui a fait le moins de jobs et le nombre de jobs faits """
        min_jobs = np.min(np.sum(self.job_done[:, np.newaxis] * self.x, axis=(0, 1)))
        worker_index = np.argmin(np.sum(self.job_done[:, np.newaxis] * self.x, axis=(0, 1)))
        return worker_index, min_jobs

    def compute_objective_values(self, ponderation_task_done="one"):
        """
        precondition: pondération des task done est le niveau de la task 
        skills: somme des niveaux
        cognitive load: somme des charges cognitives
        """
        benefit = 0
        for i in range(len(self.instance.jobs_struct)):
            benefit += self.instance.resale_price_jobs[i] * self.job_done[i]
            for j in range (len(self.instance.jobs_struct[i])):
                if ponderation_task_done == "one":
                    benefit += self.task_done[i,j]
                elif ponderation_task_done == "difficulty":
                    benefit += self.task_done[i,j] * self.instance.get_difficulty_and_metier_of_task(i,j)[0]

        skills = np.sum(self.l) - np.sum(self.instance.levels_workers)
        print("skills", skills)
        cognitive_load = - np.sum(self.cognitive_load_total)
        self.objective_values = {"0": benefit, "1": skills, "2": cognitive_load}
        return self.objective_values


    def print_objective_values(self):
        print("Objective values:")
        for key, value in self.objective_values.items():
            print(f"Obj{key}: {value}")
        for key, value in self.objective_values.items():
            if key == "0":
                print(f"Benefit: {value}")
            elif key == "1":
                print(f"Skills: {value}")
            elif key == "2":
                print(f"Cognitive load: {value}")


    def get_summary(self, label=""):
        """Retourne un dict de métriques résumant la solution."""
        nb_jobs_total = self.instance.nb_jobs
        nb_ops_total = sum(len(self.instance.jobs_struct[i]) for i in range(nb_jobs_total))
        nb_jobs_done = int(np.sum(self.job_done))
        nb_ops_done = int(np.sum(self.task_done))

        mode_counts = [0, 0, 0, 0]
        for i in range(nb_jobs_total):
            for j in range(len(self.instance.jobs_struct[i])):
                for mode in range(4):
                    if self.z_auxilary[i, j, mode] > 0.5:
                        mode_counts[mode] += 1

        benefit = None # correspond a la valeur de l'objectif pas au benefice de job done
        if self.objective_values:
            benefit = float(self.objective_values.get("0", self.objective_values.get(0, 0)))

        nb_ops_production = mode_counts[0] + mode_counts[2]        # solo + collab
        nb_ops_formation  = mode_counts[1] + mode_counts[3]        # teaching + solo sans niveau

        return {
            "label": label,
            "nb_jobs_total": nb_jobs_total,
            "nb_jobs_done": nb_jobs_done,
            "nb_ops_total": nb_ops_total,
            "nb_ops_done": nb_ops_done,
            "mode_solo": mode_counts[0],
            "mode_teaching": mode_counts[1],
            "mode_collab": mode_counts[2],
            "mode_solo_no_level": mode_counts[3],
            "nb_ops_production": nb_ops_production,
            "nb_ops_formation": nb_ops_formation,
            "nb_ops_with_level": mode_counts[0] + mode_counts[1] + mode_counts[2],
            "nb_ops_without_level": mode_counts[3],
            "total_penalty_levels": float(np.sum(self.penalty_levels)),
            "benefit": benefit,
            "skill_gain": float(self.sum_skill_levels_rate()),
            "cognitive_load": float(self.sum_cognitive_load_total()),
        }

    @staticmethod
    def plot_comparison(summaries, title="Comparaison des solutions"):
        """Affiche un graphique Plotly comparant plusieurs résumés de solutions.

        Args:
            summaries: liste de dicts retournés par get_summary()
            title: titre global du graphique
        """
        labels = [s["label"] or f"Solution {i}" for i, s in enumerate(summaries)]
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            subplot_titles=(
                "Jobs et opérations terminés",
                "Modes d'opération",
                "Bilan formation vs. production",
                "Qualité des opérations",
                "Objectifs",
                "",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.12,
        )

        # --- [1,1] Jobs / ops terminés ---
        scheduling_metrics = [
            ("Jobs terminés", [s["nb_jobs_done"]  for s in summaries]),
            ("Jobs total",    [s["nb_jobs_total"]  for s in summaries]),
            ("Ops terminées", [s["nb_ops_done"]    for s in summaries]),
            ("Ops total",     [s["nb_ops_total"]   for s in summaries]),
        ]
        for idx, sol_label in enumerate(labels):
            fig.add_trace(go.Bar(
                name=sol_label,
                x=[m[0] for m in scheduling_metrics],
                y=[m[1][idx] for m in scheduling_metrics],
                marker_color=colors[idx % len(colors)],
                legendgroup=sol_label,
                showlegend=True,
                hovertemplate=f"<b>{sol_label}</b><br>%{{x}} : %{{y}}<extra></extra>",
            ), row=1, col=1)

        # --- [1,2] Modes d'opération ---
        mode_names  = ["Solo", "Teaching", "Collab", "Solo sans niveau"]
        mode_keys   = ["mode_solo", "mode_teaching", "mode_collab", "mode_solo_no_level"]
        mode_colors = ["#19D3F3", "#FF6692", "#B6E880", "#FF97FF"]
        for m_idx, (mode_name, mode_key) in enumerate(zip(mode_names, mode_keys)):
            fig.add_trace(go.Bar(
                name=mode_name,
                x=labels,
                y=[s[mode_key] for s in summaries],
                marker_color=mode_colors[m_idx],
                legendgroup=mode_name,
                showlegend=True,
                hovertemplate=f"<b>%{{x}}</b><br>{mode_name} : %{{y}} ops<extra></extra>",
            ), row=1, col=2)

        # --- [2,1] Bilan formation vs. production ---
        bilan_metrics = [
            ("Production\n(S + C)",          "nb_ops_production"),
            ("Formation\n(T + S_no_lev)",   "nb_ops_formation"),
        ]
        for idx, sol_label in enumerate(labels):
            fig.add_trace(go.Bar(
                name=sol_label,
                x=[m[0] for m in bilan_metrics],
                y=[summaries[idx].get(m[1], 0) for m in bilan_metrics],
                marker_color=colors[idx % len(colors)],
                legendgroup=sol_label,
                showlegend=False,
                text=[summaries[idx].get(m[1], 0) for m in bilan_metrics],
                textposition="outside",
                hovertemplate=f"<b>{sol_label}</b><br>%{{x}} : %{{y}} ops<extra></extra>",
            ), row=2, col=1)

        # --- [2,2] Qualité des opérations ---
        qualite_metrics = [
            ("Avec niveau requis", "nb_ops_with_level"),
            ("Sans niveau requis", "nb_ops_without_level"),
        ]
        for idx, sol_label in enumerate(labels):
            penalty = summaries[idx].get("total_penalty_levels", 0)
            fig.add_trace(go.Bar(
                name=sol_label,
                x=[m[0] for m in qualite_metrics],
                y=[summaries[idx].get(m[1], 0) for m in qualite_metrics],
                marker_color=colors[idx % len(colors)],
                legendgroup=sol_label,
                showlegend=False,
                text=[summaries[idx].get(m[1], 0) for m in qualite_metrics],
                textposition="outside",
                hovertemplate=(
                    f"<b>{sol_label}</b><br>%{{x}} : %{{y}} ops"
                    f"<br>Pénalité totale : {penalty:.2f}<extra></extra>"
                ),
            ), row=2, col=2)

        # --- [3,1] Objectifs (colspan=2) ---
        obj_metrics = [
            ("Bénéfice",          "benefit"),
            ("Gain compétences",  "skill_gain"),
            ("Charge cognitive",  "cognitive_load"),
        ]
        for idx, sol_label in enumerate(labels):
            fig.add_trace(go.Bar(
                name=sol_label,
                x=[m[0] for m in obj_metrics],
                y=[summaries[idx].get(m[1]) or 0 for m in obj_metrics],
                marker_color=colors[idx % len(colors)],
                legendgroup=sol_label,
                showlegend=False,
                hovertemplate=f"<b>{sol_label}</b><br>%{{x}} : %{{y:.2f}}<extra></extra>",
            ), row=3, col=1)

        fig.update_layout(
            title_text=title,
            barmode="group",
            height=900,
            legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        )
        fig.show()
        return fig


    def solutions_to_df(list_solutions, labels=None):
        rows = []
        for i, s in enumerate(list_solutions):
            if s is None:
                continue
            label = labels[i] if labels else f"Solution {i}"
            w_max, cog_max = s.get_cognitive_load_of_workers_highest()
            w_max_tasks, nb_max_tasks = s.get_worker_do_maximum_tasks()
            w_min_tasks, nb_min_tasks = s.get_worker_do_minimum_tasks()
            nb_jobs_no_level, _ = s.get_nb_jobs_done_with_some_task_done_solo_no_level()
            nb_jobs_any_no_level, _ = s.get_nb_jobs_with_some_task_done_solo_no_level()

            rows.append({
                "label":                       label,
                "resale_price":                s.get_resale_price_job_done(),
                "nb_jobs_done":                s.get_nb_jobs_done(),
                "nb_jobs_started_not_done":    s.get_nb_jobs_beginned_but_not_done(),
                "nb_tasks_done":               s.get_nb_tasks_done(),
                "nb_tasks_solo":               s.get_nb_tasks_done_solo(),
                "nb_tasks_teaching":           s.get_nb_tasks_done_teaching(),
                "nb_tasks_collab":             s.get_nb_tasks_done_collab(),
                "nb_tasks_solo_no_level":      s.get_nb_tasks_done_solo_no_level(),
                "nb_jobs_done_with_no_level":  nb_jobs_no_level,
                "nb_jobs_any_no_level":        nb_jobs_any_no_level,
                "skill_gain":                  s.get_sum_skill_levels_rate(),
                "cognitive_load_total":        s.get_sum_cognitive_load_total(),
                "cognitive_load_max_worker":   cog_max,
                "worker_max_tasks":            w_max_tasks,
                "nb_max_tasks":                nb_max_tasks,
                "worker_min_tasks":            w_min_tasks,
                "nb_min_tasks":                nb_min_tasks,
            })

        return pd.DataFrame(rows).set_index("label")

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
               f"borne_sup_makespan: {self.borne_sup_makespan}\n"
               f"Level_min: {self.Level_min.shape} \n{self.Level_min}\n"
               f"Delta_min: {self.Delta_min.shape} \n{self.Delta_min}\n"
               f"task_done: {self.task_done.shape} \n{self.task_done}\n"
               )
        
        return res