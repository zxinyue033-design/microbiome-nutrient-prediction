import os
from Bio import Entrez

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

Entrez.email = "zxinyue436@gmail.com"   