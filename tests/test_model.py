import numpy as np
import pytest


from first_model import Model
from Solution import Solution
from Instance import Instance
from utils import read_file



@pytest.fixture
def instance():
    res = read_file("data/data_resale_price_1.test")
    instance = Instance()
    instance.from_dictionary(res)
    return instance


class SolutionChecker:
    def __init__(self, instance):
        self.instance = instance

    def validate_solution(self, solution):
        self._max_two_workers(solution)
        self._precedence(solution)
        self._no_overlap(solution)
        self._levels(solution)
        self._same_start_finish_operations(solution)
        self._not_collaboration_if_not_possible(solution)
        return True
    
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

        for n in range(len(prec)): #nb_jobs
            for i in range(len(prec[0])): #nb_tasks
                for j in range(i+1, len(prec[0])): #nb_tasks
                    if prec[n][i][j] == 1: #task j must be done after task i

                        worker_assigned_i = np.where(x[n, i, :] == 1)[0]
                        worker_assigned_j = np.where(x[n, j, :] == 1)[0]

                        for w_i in worker_assigned_i:
                            for w_j in worker_assigned_j:
                                assert f[n][i][w_i] <= d[n][j][w_j], f"Precedence constraint violated: for job {n}, task {i} must be done before begin task {j}"

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
                    if not (f[i1][j1][k] <= d[i2][j2][k] or f[i2][j2][k] <= d[i1][j1][k]):
                        raise AssertionError(f"Overlap detected for worker {k} between tasks ({i1}, {j1}) and ({i2}, {j2})")

    def _levels(self, solution):
        """ Check if the levels constraints are respected for each mode of assignment (solo, teaching, collab, alone_no_levels) """
        x = solution.x
        z = solution.z_auxilary # variable indicating the task mode (solo, teaching, collab, alone_no_levels)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
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
                        assert skill_w < level_task, f"Alone_no_levels mode requires the worker to have a skill level less than the task difficulty for task {i},{j}"
                
                if len(workers_assigned) == 2 :
                    w1, w2 = workers_assigned
                    skill_w1 = self.instance.levels_workers[w1][metier] # skill level of worker 1 for the profession
                    skill_w2 = self.instance.levels_workers[w2][metier] # skill level of worker 2 for the profession

                    if z[i,j,1] == 1: # if the task is assigned in teaching mode
                        assert (skill_w1 >= level_task or skill_w2 >= level_task) and (skill_w1 < level_task or skill_w2 < level_task), f"Teaching mode requires one worker to have a higher skill level than the the difficulty, and one worker to have a lower skill level than the the difficulty for task {i},{j}"
                    
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
                    assert d[i][j][w1] == d[i][j][w2], f"Tasks assigned to 2 workers must start at the same date, but for task {i},{j} they start at different dates"
                    assert f[i][j][w1] == f[i][j][w2], f"Tasks assigned to 2 workers must finish at the same date, but for task {i},{j} they finish at different dates"

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

        for i in range(x.shape[0]): # for each job
            # number of tasks assigned to each worker for job n
            counted_tasks = x[i].sum(axis=0) 
            assert np.any(counted_tasks == len(self.instance.jobs_struct[i])), f"Job {i} does not have a worker assigned to all its tasks"
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

    def no_skill_job(self, solution):
        """ Check if the tasks with no required skills are assigned to workers in alone_no_levels mode the level must be at least minus one of the difficulty of the task"""
        x = solution.x
        z = solution.z_auxilary

        for i in range(len(self.instance.jobs_struct)):
            for j in range(len(self.instance.jobs_struct[i])):
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
                        assert np.abs(level_task - self.instance.levels_workers[w][metier]) >= 1, f"Tasks with no required skills must be assigned to workers with a skill level at least one less than the task difficulty for task {i},{j}"
                        assert z[i,j,3] == 1, f"Tasks with no required skills assigned to one worker must be in solo mode, but task {i},{j} is assigned is not assigned in solo mode"

        return True


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

    validator = SolutionChecker(instance)

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

    validator = SolutionChecker(instance)

    assert validator.validate_solution(solution), "Solution is not valid"
    assert validator.no_teaching_tasks(solution), "Solution contains teaching tasks but the no_teaching_tasks constraint is activated"

