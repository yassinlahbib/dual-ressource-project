import numpy as np
import pandas as pd

from first_model import Model # pour solve_and_validate_model
from test_model import ModelSolutionValidator


def get_model_validation_tests(constraints_config=None):
    """
    Retourne la liste des tests à appliquer selon la configuration.
    """
    constraints_config = constraints_config or {}

    tests = [
        ("solution_valide", "validate_solution"),
    ]

    if not constraints_config.get("no_base_worker", False):
        tests.append(("base_worker", "base_worker"))

    if constraints_config.get("no_teaching_tasks", False):
        tests.append(("no_teaching_tasks", "no_teaching_tasks"))

    if constraints_config.get("job_with_no_skills", False):
        tests.append(("solo_no_level_difference", "solo_no_level_difference"))
    else:
        tests.append(("need_level_if_solo", "need_level_if_solo"))

    return tests


def validate_model_solution(instance, solution, constraints_config=None, name="solution"):
    
    """
    Valide une solution MILP et retourne :
    - une ligne de résultat sous forme de dict
    - un booléen global OK
    """
    validator = ModelSolutionValidator(instance, constraints_config=constraints_config)
    tests = get_model_validation_tests(constraints_config)

    row = {
        "Nom": name,
        "C_max": getattr(solution, "C_max", None),
        "Profit": solution._resale_price_job_done_plus_task_done(ponderation_task_done="one"),
        "Profit_job_done": solution.get_resale_price_job_done(),
        "Skill_gain": solution.get_sum_skill_levels_rate(),
        "Cognitive_load": solution.get_sum_cognitive_load_total(),
    }

    run_ok = True

    for test_name, method_name in tests:
        print(f"Exécution du test {test_name}...")
        try:
            method = getattr(validator, method_name)
            method(solution)
            row[test_name] = "PASS"

        
        except AssertionError as e:
            print("11")
            row[test_name] = f"FAIL {str(e)[:100]}"
            run_ok = False

        
        except Exception as e:
            print(f"Erreur inattendue lors de l'exécution du test {test_name}: {str(e)}")
            row[test_name] = f"ERR {str(e)}"
            run_ok = False

    row["OK"] = "TRUE" if run_ok else "FALSE"

    return row, run_ok


def validate_model_solutions(instance, solutions, constraint_configs=None):
    """
    Valide une ou plusieurs solution, issu de la même instance

    solutions peut être :
    - une liste de Solution
    - un dictionnaire {"nom": solution}
    """
    rows = []
    total = 0
    passed = 0

    if isinstance(solutions, dict):
        iterable = solutions.items()
    else:
        iterable = [(f"solution_{i}", sol) for i, sol in enumerate(solutions)]

    for (name, solution), config in zip(iterable, constraint_configs):
        row, ok = validate_model_solution(
            instance=instance,
            solution=solution,
            constraints_config=config,
            name=name,
        )

        rows.append(row)
        total += 1

        if ok:
            passed += 1

    df = pd.DataFrame(rows)

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "all_passed": passed == total,
    }

    return df, summary


def solve_and_validate_model(
    instance,
    objective="three",
    weight=None,
    priority=None,
    constraints_config=None,
    verbose=False,
    time_limit=None,
    objective_terms_config=None,
    benefit_in_time=True,
    name="MILP",
):
    """
    Résout le modèle puis valide automatiquement la solution.
    Pratique pour un notebook.
    """

    if weight is None:
        weight = {
            "profit": 100,
            "skills": 10,
            "cognitive_load": 1,
        }

    if priority is None:
        priority = [0, 0, 0]

    model = Model(instance)

    solution = model.solve(
        objective=objective,
        weight=weight,
        priority=priority,
        constraints_config=constraints_config,
        verbose=verbose,
        time_limit=time_limit,
        objective_terms_config=objective_terms_config,
        benefit_in_time=benefit_in_time,
    )

    if solution is None:
        df = pd.DataFrame([{
            "Nom": name,
            "OK": "FALSE",
            "Erreur": "Aucune solution retournée par le solveur",
        }])

        summary = {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "all_passed": False,
        }

        return solution, df, summary

    df, summary = validate_model_solutions(
        instance=instance,
        solutions={name: solution},
        constraints_config=constraints_config,
    )

    return solution, df, summary


def print_validation_summary(summary):
    """
    Affiche un résumé simple.
    """
    print(f"\n{'=' * 50}")

    if summary["all_passed"]:
        print("TOUS LES TESTS PASSENT")
    else:
        print("CERTAINS TESTS ÉCHOUENT")

    print(f"{summary['passed']}/{summary['total']} solution(s) valide(s)")