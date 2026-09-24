# Dual-Resource Constrained Scheduling Problem


---

## Problème

Ordonnancement de jobs avec **double contrainte de ressource** :

- Chaque job est composé d'opérations à affecter à des travailleurs
- Les travailleurs ont des niveaux de compétence par métier (1 à 4)
- Une opération peut être réalisée en **solo**, en **tutorat** ou **collaboration** ou en **solo no level (difference de niveau de 1)**  
- Objectifs multiples : maximiser le bénéfice (prix de revente des jobs complétés), minimiser le makespan et la charge de cognitive des travailleurs du aux tutorats et collaborations.

---

## Structure du projet

```
src/
  Instance.py               — Chargement et génération d'instances
  Solution.py               — Représentation d'une solution (x, d, C, C_max, ...)
  first_model.py            — Modèle MILP (Gurobi)
  metaheuristic_approach.py — Approche métaheuristique
  utils.py                  — Fonctions utilitaires et visualisation
  Constante.py              — Constantes globales du modèle

data/                       — Instances de test (.test)
results/                    — Fichiers de sortie (solutions Gurobi, Gantt)
notebook/                   — Notebooks d'expérimentation et de comparaison
tests/                      — Tests unitaires (pytest)
doc/                        — Documentation
```

---

## Approches implémentées

### 1. MILP (Gurobi)
Modèle en programmation linéaire mixte en nombres entiers avec 3 objectifs hiérarchiques.  

### 2. Métaheuristique
Encodage par chromosome, recherche locale, voisinages multiples.  

---

## Installation

```bash
pip install gurobipy numpy pandas matplotlib plotly tqdm pytest
```

Gurobi nécessite une licence valide (licence académique disponible sur [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)).

---

## Utilisation

```bash
cd src/

# Résoudre une instance avec le MILP
python first_model.py

# Lancer la métaheuristique
python metaheuristic_approach.py

# Tests unitaires
cd ..
pytest tests/
```

---

## Format des instances

Les fichiers `.test` dans `data/` suivent ce format :

```
<number of jobs>
<maximal number of operations>
<number of workers>
<levels of workers>          — niveau par métier pour chaque travailleur
<difficulty of jobs>
<(processing time, difficulty) of operations>
<precedence constraints of operations>
```

---

## Résultats

Les résultats sont sauvegardés dans `results/` :
- `solution.sol` — solution finale Gurobi
- `intermediate_solutions.sol` — solutions intermédiaires pendant l'optimisation
- `model.lp` — formulation LP du modèle
- `gantt_chart.html` — diagramme de Gantt interactif (Plotly)
