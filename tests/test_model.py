import numpy as np
import pytest


from first_model import Model
from Solution import Solution
from Instance import Instance
from utils import read_file
from Constante import PERC_SOLO_NO_LEVEL_TIME, LEVEL_MAX



@pytest.fixture
def instance():
    res = read_file("data/data_resale_price_1.test")
    instance = Instance()
    instance.from_dictionary(res)
    return instance


class ModelSolutionValidator:
    def __init__(self, instance, constraints_config=None):
        self.instance = instance
        self.constraints_config = constraints_config
        self.nb_job_no_skills = 1 if self.constraints_config.get("job_with_no_skills", False) else 0

    def validate_solution(self, solution):
        self._max_two_workers(solution)
        self._precedence(solution)
        self._no_overlap(solution)
        self._levels(solution)
        self._same_start_finish_operations(solution)
        self._not_collaboration_if_not_possible(solution)
        self._job_done_consistent_with_x(solution)
        self._durees_coherentes_avec_mode(solution)
        self._cmax_coherent(solution)
        self._taches_non_assignees_inactives(solution)
        self._mode_z_coherent_avec_x(solution)
        self._niveaux_non_decroissants(solution)
        self._charge_cognitive_coherente(solution)
        self._solo_no_level_au_plus_n_jobs(solution)
        self._profit_coherent_avec_job_done_plus_task_done(solution)
        self._check_task_done_affectation_variable(solution)
        self._job_done_consistent_with_task_done(solution)

        return True
    
    #======== contraintes physiques

    def _max_two_workers(self, solution):
        """ Check if no task is assigned to more than 2 workers """
        x = solution.x
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                assert sum(x[i, j, :]) <= 2, f"More than 2 workers assigned to task {i},{j}"

    def _precedence(self, solution):
        """ Check if the precedence constraints are respected """
        d = solution.d
        f = solution.f
        x = solution.x
        prec = self.instance.constraints_precedence_operations
        # print(prec)

        for n in range(len(self.instance.jobs_struct)): #nb_jobs
            for i in range(len(self.instance.jobs_struct[n])): #nb_tasks
                for j in range(i+1, (len(self.instance.jobs_struct[n]))): #nb_tasks
                    if prec[n][i][j] == 1: #task j must be done after task i

                        worker_assigned_i = np.where(x[n, i, :] == 1)[0]
                        worker_assigned_j = np.where(x[n, j, :] == 1)[0]

                        if len(worker_assigned_j) > 0 and len(worker_assigned_i) == 0:
                            raise AssertionError(
                                f"O_{n},{j} done but its predecessor O_{n},{i} is not"
                            )

                        for w_i in worker_assigned_i:
                            for w_j in worker_assigned_j:
                                assert f[n][i][w_i] <= d[n][j][w_j]+0.01, f"{n}{i}{j}{w_i}{w_j}:{f[n][i][w_i]:.2f}>{d[n][j][w_j]:.2f}:PCV: O_{n},{i} must be < {j}"

    def _no_overlap(self, solution):
        """ Check if there is no overlap between tasks assigned to the same worker """
        d = solution.d
        f = solution.f
        x = solution.x
        nb_workers = self.instance.nb_workers
        
        for k in range(nb_workers):

            tasks_assigned = []
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i, j, k] == 1:
                        tasks_assigned.append((i, j))

            for idx1 in range(len(tasks_assigned)):
                for idx2 in range(idx1 + 1, len(tasks_assigned)):
                    i1, j1 = tasks_assigned[idx1]
                    i2, j2 = tasks_assigned[idx2]

                    # Si un se finit avant le debut de l'autre, il n'y a pas de chevauchement -> prendre négation pour détecter le chevauchement
                    if not (f[i1][j1][k] <= d[i2][j2][k]+0.01 or f[i2][j2][k] <= d[i1][j1][k]+0.01):
                        raise AssertionError(f"Overlap w{k} between op ({i1}, {j1}) and ({i2}, {j2})")

    def _check_task_done_affectation_variable(self, solution):
        """ check que task_done vaut bien 1 si au moins 1 worker dessus et 0 si pas de worker dessus """
        task_done = solution.task_done
        x = solution.x

        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                nb_worker_assigned = np.sum(x[i, j, :])
                if nb_worker_assigned > 0:
                    assert task_done[i, j] == 1, f"Task {i},{j} is assigned to a worker but task_done is 0"
                else:
                    assert task_done[i, j] == 0, f"Task {i},{j} is not assigned to any worker but task_done is 1"

    #======== Modes et niveaux

    def _levels(self, solution):
        """ Check if the levels constraints are respected for each mode of assignment (solo, teaching, collab, alone_no_levels) """
        x = solution.x
        z = solution.z_auxilary # variable indicating the task mode (solo, teaching, collab, alone_no_levels)

        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                if len(workers_assigned) == 0:
                    assert np.sum(z[i,j,:]) == 0, f"Task {i},{j} is not assigned to any worker but has a mode assigned in z"
                else:
                    assert np.sum(z[i,j,:]) == 1, f"Task {i},{j} must be assigned to exactly one mode (solo, teaching, collab, alone_no_levels)"
                
                
                if len(workers_assigned) > 0 : # if task is assigned to at least one worker
                    task_id_instance = self.instance.jobs_struct[i][j]
                    metier = self.instance.task_to_m[task_id_instance] # get the profession of the task
                    level_task = self.instance.tasks_difficulties[task_id_instance] # get the level difficulty of the task

                if len(workers_assigned) == 1 :
                    w = workers_assigned[0]
                    skill_w = self.instance.levels_workers[w][metier] # skill level of the worker for the profession

                    if z[i,j,0] == 1: # if the task is assigned in solo mode
                        assert skill_w >= level_task, f"Solo mode requires the worker to have a skill level more than the task difficulty for task {i},{j}"
                    
                    elif z[i,j,3] == 1: # if the task is assigned in alone_no_levels mode
                        assert (skill_w < level_task) and (skill_w + 1) >= level_task, f"Alone_no_levels mode requires the worker to have a skill level less than the task difficulty for task {i},{j}"
                
                if len(workers_assigned) == 2 :
                    w1, w2 = workers_assigned
                    skill_w1 = self.instance.levels_workers[w1][metier] # skill level of worker 1 for the profession
                    skill_w2 = self.instance.levels_workers[w2][metier] # skill level of worker 2 for the profession

                    if z[i,j,1] == 1: # if the task is assigned in teaching mode
                        assert (skill_w1 >= level_task or skill_w2 >= level_task) and (skill_w1 < level_task or skill_w2 < level_task), f"Teaching mode requires one worker to have a higher skill level than the the difficulty, and one worker to have a lower skill level than the the difficulty for task {i},{j}"
                        assert (skill_w1 + 1) >= level_task and (skill_w2 + 1) >= level_task, f"Tech mode requires both workers to have a skill level at most 1 less than the task difficulty for task {i},{j}"

                    elif z[i,j,2] == 1: # if the task is assigned in collaboration mode
                        assert skill_w1 >= level_task and skill_w2 >= level_task, f"Collaboration mode requires both workers to have the skill level more than the task difficulty for task {i},{j}"

    def _same_start_finish_operations(self, solution):
        """ Check if an operation are assigned to 2 workers, it must be start at the same date """
        x = solution.x
        d = solution.d
        f = solution.f

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                if len(workers_assigned) == 2 : # if task is assigned to 2 workers
                    w1, w2 = workers_assigned
                    assert abs(d[i][j][w1] - d[i][j][w2]) < 0.01, f"Tasks assigned to 2 workers must start at the same date, but for task {i},{j} they start at different dates"
                    assert abs(f[i][j][w1] - f[i][j][w2]) < 0.01, f"Tasks assigned to 2 workers must finish at the same date, but for task {i},{j} they finish at different dates"

    def _not_collaboration_if_not_possible(self, solution):
        """ Check if a task is assigned to 2 workersbut in the time variable to the instance, collaborative_task is -1, then the task cannot be assigned in collaboration mode """
        
        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                workers_assigned = np.where(solution.x[i, j, :] == 1)[0]

                if len(workers_assigned) == 2 :  # if task is assigned to 2 workers
                    w1, w2 = workers_assigned
                    task_id_instance = self.instance.jobs_struct[i][j]
                    metier = self.instance.task_to_m[task_id_instance] # get the profession of the task
                    l_w1 = self.instance.levels_workers[w1][metier] # skill level of worker 1 for the profession
                    l_w2 = self.instance.levels_workers[w2][metier] # skill level of worker 2 for the profession
                    task_difficulty = self.instance.tasks_difficulties[task_id_instance] # difficulty level of the task
                    
                    if  self.instance.tasks_times[task_id_instance][2] == -1: # if the task cannot be assigned in collaboration mode according to the instance data
                        assert solution.z_auxilary[i,j,2] == 0, f"Task {i},{j} is assigned to 2 workers but cannot be assigned in collaboration mode according to the instance data"
                        assert l_w1 >= task_difficulty or l_w2 >= task_difficulty
                        assert l_w1 < task_difficulty or l_w2 < task_difficulty

    #======= Cohérence job_done / assignment

    def _job_done_consistent_with_x(self, solution):
        """ Check if the job_done variable is consistent with the assignment of tasks to workers """
        x = solution.x
        job_done = solution.job_done

        for i in range(self.instance.nb_jobs):
            # print(f"Checking job_done consistency for job {i}")
            # print("job_done:", job_done[i])
            all_operations_assigned = True

            for j in range(len(self.instance.jobs_struct[i])): 
                if np.sum(x[i, j, :]) == 0:
                    all_operations_assigned = False
                    break
            # print("job_done_expected:", job_done_expected)

            job_done_expected = 1 if all_operations_assigned else 0
            # print(f"Job {i}: job_done={job_done[i]}, expected={job_done_expected}")
            assert job_done[i] == job_done_expected, f"Job {i} is marked as done but not all its tasks are assigned to workers"

    def _job_done_consistent_with_task_done(self, solution):
        """ Check if the job_done variable is consistent with the task_done variable """
        task_done = solution.task_done
        job_done = solution.job_done

        for i in range(self.instance.nb_jobs):
            all_operations_done = True

            for j in range(len(self.instance.jobs_struct[i])): 
                if task_done[i, j] == 0:
                    all_operations_done = False
                    break

            job_done_expected = 1 if all_operations_done else 0
            assert job_done[i] == job_done_expected, f"Job {i} is marked as done but not all its tasks are marked as done"

    #======= Validation optionnelle des contraintes spécifiques

    def no_teaching_tasks(self, solution):
        """ Check if there is no teaching tasks in the solution (if the no_teaching_tasks constraint is activated) """
        z = solution.z_auxilary # variable indicating the task mode (solo, teaching, collab, alone_no_levels)
        x = solution.x

        # Check for the z variables corresponding to the teaching mode
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                assert z[i,j,1] == 0, f"Teaching mode is not allowed but task {i},{j} is assigned in teaching mode"

        # Check for the x variables corresponding to 2 workers assigned to the same task (which would indicate a teaching or collaboration mode)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                workers_assigned = np.where(x[i, j, :] == 1)[0]


                 # if 2 workers are assigned to the same task, it can only be in collaboration mode (if the no_teaching_tasks constraint is activated)
                if len(workers_assigned) == 2 :
                    task_id_instance = self.instance.jobs_struct[i][j]
                    metier = self.instance.task_to_m[task_id_instance] # get the profession of the task
                    level_task = self.instance.tasks_difficulties[task_id_instance] # get the level difficulty of the task

                    w1, w2 = workers_assigned
                    skill_w1 = self.instance.levels_workers[w1][metier] # skill level of worker 1 for the profession
                    skill_w2 = self.instance.levels_workers[w2][metier] # skill level of worker 2 for the profession

                    assert z[i,j,1] == 0, f"Teaching mode is not allowed but task {i},{j} is assigned in teaching mode"
                    assert skill_w1 >= level_task and skill_w2 >= level_task, f"Collaboration mode requires both workers to have the skill level more than the task difficulty for task {i},{j}"                            
        return True

    def base_worker(self, solution):
        """ Check if at least one worker follows all the tasks of the same job (if the no_base_worker constraint is not activated) """
        x = solution.x
        y = solution.y

        for i in range(len(self.instance.jobs_struct)): # for each job
            if solution.job_done[i] == 1: # if the job is done
                w = np.where(y[i, :] == 1)[0] # get the worker assigned as base worker for job i
                assert np.sum(x[i,:,w]) == len(self.instance.jobs_struct[i]), f"Job {i} is marked as done but the base worker {w} is not assigned to all its tasks"
            else :
                nb_task_done = np.sum(solution.task_done[i, :])
                w = np.where(y[i, :] == 1)[0] # get the worker assigned as base worker for job i
                assert np.sum(x[i,:,w]) == nb_task_done, f"J_{i} is not marked as done but the base worker {w} is not assigned to all its tasks that are done"


        return True

    def need_level_if_solo(self, solution):
        """ Check if the worker assigned to a task in solo mode has the required skill level for the task (if the job_with_no_skills constraint is activated) """
        x = solution.x
        z = solution.z_auxilary

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                if len(workers_assigned) == 1 and z[i,j,0] == 1: # if the task is assigned to exactly one worker and in solo mode
                    w = workers_assigned[0]
                    task_id_instance = self.instance.jobs_struct[i][j]
                    metier = self.instance.task_to_m[task_id_instance] # get the profession of the task
                    level_task = self.instance.tasks_difficulties[task_id_instance] # get the level difficulty of the task
                    skill_w = self.instance.levels_workers[w][metier] # skill level of the worker for the profession

                    assert skill_w >= level_task, f"Solo mode requires the worker to have a skill level more than the task difficulty for task {i},{j}"

        return True

    def solo_no_level_difference(self, solution, max_difference=1):
        """ Check if the tasks with no required skills are assigned to workers in alone_no_levels mode the level must be at least minus one of the difficulty of the task"""
        # print("here")
        x = solution.x
        z = solution.z_auxilary

        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                # print(f"Checking task {i},{j} for solo_no_level_difference...")
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                if len(workers_assigned) == 1 : # if task is assigned to at least one worker
                    w = workers_assigned[0]
                    task_id_instance = self.instance.jobs_struct[i][j]
                    metier = self.instance.task_to_m[task_id_instance] # get the profession of the task
                    level_task = self.instance.tasks_difficulties[task_id_instance] # get the level difficulty of the task

                    # LEVEL
                    if self.instance.levels_workers[w][metier] >= level_task: # if the worker has the required skill level for the task
                        assert z[i,j,0] == 1, f"Tasks with required skills must be assigned in solo mode for task {i},{j} "
                    
                    # NO LEVEL
                    if self.instance.levels_workers[w][metier] < level_task: # if the worker does not have the required skill level for the task
                        assert np.abs(level_task - self.instance.levels_workers[w][metier]) <= max_difference, f" workers {w} with a skill level {self.instance.levels_workers[w][metier]} can not do task {i},{j} because level is {level_task} "
                        assert z[i,j,3] == 1, f"Tasks with no required skills assigned to one worker must be in solo mode, but task {i},{j} is assigned is not assigned in solo mode"
        return True



    def _durees_coherentes_avec_mode(self, solution):
        """Durée d'exécution = temps du mode utilisé (solo/teaching/collab/alone_no_level)"""
        x = solution.x
        d = solution.d
        f = solution.f
        z = solution.z_auxilary

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                if len(workers_assigned) == 0:
                    continue

                task_id = self.instance.jobs_struct[i][j]

                if z[i, j, 0] == 1:
                    expected_time = self.instance.tasks_times[task_id][0]
                    mode_name = "solo"
                elif z[i, j, 1] == 1:
                    expected_time = self.instance.tasks_times[task_id][1]
                    mode_name = "teaching"
                elif z[i, j, 2] == 1:
                    expected_time = self.instance.tasks_times[task_id][2]
                    mode_name = "collab"
                elif z[i, j, 3] == 1:
                    expected_time = self.instance.tasks_times[task_id][0] * PERC_SOLO_NO_LEVEL_TIME
                    mode_name = "alone_no_level"
                else:
                    raise AssertionError(
                        f"Opération ({i},{j}) assignée mais aucun mode z actif"
                    )

                for w in workers_assigned:
                    actual_time = f[i][j][w] - d[i][j][w]
                    assert abs(actual_time - expected_time) < 1e-6, (
                        f"Opération ({i},{j}) worker {w} mode={mode_name} : "
                        f"durée={actual_time:.2f} attendue={expected_time:.2f}"
                    )

    def _cmax_coherent(self, solution):
        """C_max doit être sup ou égal au maximum des dates de fin de toutes les opérations assignées"""
        x = solution.x
        f = solution.f

        max_finish = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for w in range(x.shape[2]):
                    if x[i, j, w] == 1:
                        max_finish = max(max_finish, f[i][j][w])

        assert solution.C_max >= max_finish-1e-6, (
            f"C_max={solution.C_max:.2f} < max des dates de fin={max_finish:.2f}"
        )

    def _taches_non_assignees_inactives(self, solution):
        """Si x[i,j,w] == 0 alors d[i,j,w] == 0 et f[i,j,w] == 0."""
        x = solution.x
        d = solution.d
        f = solution.f

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for w in range(x.shape[2]):
                    if x[i, j, w] == 0:
                        assert d[i][j][w] == 0, (
                            f"d: w_{w} non assigné à ({i},{j}) mais d={d[i][j][w]:.2f} != 0"
                        )
                        # Si woker k ne fait pas O_ij alors x_ijk = 0 et ne pas prendre en compte f_ijk
                        # assert f[i][j][w] == 0, (
                        #     f"f: w_{w} non assigné à ({i},{j}) mais f={f[i][j][w]:.2f} != 0"
                        # )

    def _mode_z_coherent_avec_x(self, solution):
        """Le mode z est strictement déterminé par le nombre de workers affectés et leurs niveaux"""
        x = solution.x
        z = solution.z_auxilary

        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
                workers_assigned = np.where(x[i, j, :] == 1)[0]
                nb_workers = len(workers_assigned)

                task_id = self.instance.jobs_struct[i][j]
                metier = self.instance.task_to_m[task_id]
                level_task = self.instance.tasks_difficulties[task_id]

                

                if nb_workers == 0:
                    assert np.sum(z[i, j, :]) == 0, (
                        f"Opération ({i},{j}) sans worker assigné mais z={z[i,j,:]} non nul"
                    )
                elif nb_workers == 1:
                    w = workers_assigned[0]
                    skill = self.instance.levels_workers[w][metier]
                    if skill >= level_task:
                        assert z[i, j, 0] == 1 and z[i, j, 3] == 0, (
                            f"Opération ({i},{j}) solo avec niveau : z[0] doit être 1, "
                            f"obtenu z={z[i,j,:]}"
                        )
                    else:
                        assert z[i, j, 3] == 1 and z[i, j, 0] == 0, (
                            f"Opération ({i},{j}) solo sans niveau : z[3] doit être 1, "
                            f"obtenu z={z[i,j,:]}"
                        )
                elif nb_workers == 2:
                    w1, w2 = workers_assigned
                    s1 = self.instance.levels_workers[w1][metier]
                    s2 = self.instance.levels_workers[w2][metier]
                    if s1 >= level_task and s2 >= level_task:
                        assert z[i, j, 2] == 1, (
                            f"Opération ({i},{j}) 2 workers qualifiés → collab attendue "
                            f"(z[2]=1), obtenu z={z[i,j,:]}"
                        )
                    else:
                        assert z[i, j, 1] == 1, (
                            f"Opération ({i},{j}) enseignement attendu (z[1]=1), "
                            f"obtenu z={z[i,j,:]}"
                        )

    def _niveaux_non_decroissants(self, solution):
        """Aucun worker ne doit perdre de compétences après exécution du planning"""
        l_after  = solution.l
        l_before = self.instance.levels_workers

        for k in range(self.instance.nb_workers):
            for m in range(self.instance.nb_professions):

                assert l_after[k, m] <= l_before[k, m] + 1, (
                    f"w{k},m{m} :  level_avant={l_before[k,m]:.2f}, "
                    f"après={l_after[k,m]:.2f} — gain de compétence > 1 détecté"
                )

                assert l_after[k, m] <= LEVEL_MAX, (
                    f"w{k},m{m} :  level_avant={l_before[k,m]:.2f}, "
                    f"après={l_after[k,m]:.2f} — gain de compétence > LEVEL_MAX={LEVEL_MAX} détecté"
                )

                assert l_after[k, m]+0.001 >= l_before[k, m], (
                    f"w{k},m{m} :  level_avant={l_before[k,m]:.2f}, "
                    f"après={l_after[k,m]:.2f} — perte de compétence détectée"
                )

    def _charge_cognitive_coherente(self, solution):
        """cognitive_load_total doit être la somme de ses composantes et être >= 0"""
        for k in range(self.instance.nb_workers):
            for m in range(self.instance.nb_professions):
                expected = (
                    solution.cognitive_load_tutors[k, m]
                    + solution.cognitive_load_apprentis[k, m]
                    + solution.cognitive_load_collaboration[k, m]
                )
                actual = solution.cognitive_load_total[k, m]
                assert abs(actual - expected) < 1e-6, (
                    f"w_{k} m:{m} : clt={actual:.4f} "
                    f"!= t+a+c={expected:.4f}"
                )
                assert actual >= 0, (
                    f"Worker {k} métier {m} : charge cognitive négative ({actual:.4f})"
                )

    def _solo_no_level_au_plus_n_jobs(self, solution):
        """Le mode alone_no_level (z[3]) ne peut concerner qu'au plus nb_job_no_skills jobs."""
        z = solution.z_auxilary
        jobs_with_alone_no_level = set()
        print(f"nb_job_no_skills={self.nb_job_no_skills}")

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i, j, 3] == 1:
                    jobs_with_alone_no_level.add(i)

        assert len(jobs_with_alone_no_level) <= self.nb_job_no_skills, (
            f"a_n_l use for {len(jobs_with_alone_no_level)} job "
            f"{jobs_with_alone_no_level} — alowed : {self.nb_job_no_skills}"
        )

    def _profit_coherent_avec_job_done_plus_task_done(self, solution, ponderation="one"):
        """_resale_price_job_done_plus_task_done() == somme des prix des jobs dont job_done[i] == 1."""
        expected = sum(
            self.instance.resale_price_jobs[i]
            for i in range(self.instance.nb_jobs)
            if solution.job_done[i] == 1
        )
        if ponderation == "difficulty":
            expected += sum(
                solution.task_done[i, j] * self.instance.get_difficulty_and_metier_of_task(i,j)[0]
                for i in range(self.instance.nb_jobs)
                for j in range(len(self.instance.jobs_struct[i]))
            )
        elif ponderation == "one":
            expected += sum(
                solution.task_done[i, j]
                for i in range(self.instance.nb_jobs)
                for j in range(len(self.instance.jobs_struct[i]))
            )
        else:
            raise ValueError(f"Invalid ponderation value: {ponderation}")
        
        actual = solution._resale_price_job_done_plus_task_done(ponderation_task_done=ponderation)
        assert abs(actual - expected) < 1e-6, (
            f"Profit calculé={actual:.2f} != somme jobs faits={expected:.2f}"
        )


def test_solution(instance):
    
    TIME_LIMIT_MAKESPAN = instance.sum_of_job_alone() / 2

    constraints_config = {"job_with_no_skills": False, 
                          "no_teaching_tasks": False, 
                          "constrained_makespan" : TIME_LIMIT_MAKESPAN,
                          "no_base_worker": False
    }

    weights = {"profit" : 100, 
               "skills" : 10, 
               "cognitive_load" : 1
    }

    model = Model(instance)
    solution = model.solve(objective="three", 
                           weight=weights, 
                           priority=[0, 0, 0], 
                           constraints_config=constraints_config, 
                           verbose=False
    )

    validator = ModelSolutionValidator(instance, constraints_config=constraints_config)

    assert validator.validate_solution(solution), "Solution is not valid"
    assert validator.base_worker(solution), "Solution does not respect the base worker constraint"
    assert validator.need_level_if_solo(solution), "Solution does not respect the skill level requirement for solo mode constraint"


def test_no_teaching_tasks(instance):
    
    TIME_LIMIT_MAKESPAN = instance.sum_of_job_alone() / 2

    constraints_config = {"job_with_no_skills": False, 
                          "no_teaching_tasks": True, 
                          "constrained_makespan" : TIME_LIMIT_MAKESPAN,
                          "no_base_worker": False
    }

    weights = {"profit" : 100, 
               "skills" : 10, 
               "cognitive_load" : 1
    }

    model = Model(instance)
    solution = model.solve(objective="three", 
                           weight=weights, 
                           priority=[0, 0, 0], 
                           constraints_config=constraints_config, 
                           verbose=False
    )

    validator = ModelSolutionValidator(instance, constraints_config=constraints_config)

    assert validator.validate_solution(solution), "Solution is not valid"
    assert validator.no_teaching_tasks(solution), "Solution contains teaching tasks but the no_teaching_tasks constraint is activated"
