import gurobipy as gp
from gurobipy import GRB
from utils import *
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from Instance import *
from Solution import *
from Constante import *
from first_model import *

print("======== INSTANCE ========")
# res = read_file("../data/data_resale_price_simple.test")
res = read_file("../data/data_resale_price_1.test")
instance = Instance()
instance.from_dictionary(res)

print(instance.sum_of_job_alone())
print(instance)
print(instance.qualified_workers_for_task(verbose=True))
print(instance.feasible_jobs())

print("======== MODEL ========")
model = Model(instance)


print("======== CONFIG ========")
constraints_config = {"constrained_makespan": instance.sum_of_job_alone() /2,
                      "job_with_no_skills": True }

# de base : 
# - worker de base 
# - pas le droit de fair une opération sans level requis


print("======== SOLVE ========")
# s = model.solve(objective="lexicographic", weight=[0, 0, 0], priority=[2, 1, 0], constraints_config=constraints_config, verbose=False)
s = model.solve(objective="benefit", weight=[0, 0, 0], priority=[2, 1, 0], constraints_config=constraints_config, verbose=False)


print("======== RESULTS ========")
df = scheduling_to_df(s, instance)
gantt_chart(df, render="html", separate_little=True, color=3)
print(df)
print(check_df(df))
print(s.all_jobs_completed())
print(s.which_jobs_are_completed())
print(s.penalty_makespan)
print(s)


print("======= VISUALIZATION ========")
# resume_levels_workers(s, instance, verbose=True)
# bars_cognitive_load_total(s, instance, verbose=True)
# plot_cognitive_load_total(s, instance)

# TESTING DF FUNCTION
print(df.groupby("Operation"))


# NEW FUNCTION IN ELABORATION
visualization_before_scheduling(instance)
visualization_after_scheduling(instance, s, df)