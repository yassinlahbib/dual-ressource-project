import sys
import os

# Ajoute src au chemin pour les imports
sys.path.insert(0, "src")

# Force Python à chercher les fichiers depuis la racine du projet
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))