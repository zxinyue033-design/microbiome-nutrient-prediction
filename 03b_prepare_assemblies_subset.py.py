import os
import pandas as pd
from config import DATA_DIR

IN_MAP = os.path.join(DATA_DIR, "glucose_species_to_assembly.csv")
OUT_TXT = os.path.join(DATA_DIR, "glucose_assemblies_subset_50.txt")

df = pd.read_csv(IN_MAP)

col = "assembly_accession" if "assembly_accession" in df.columns else "assembly"

assemblies = (
    df[col].dropna().astype(str).str.strip().drop_duplicates().head(50).tolist()
)

with open(OUT_TXT, "w") as f:
    for a in assemblies:
        f.write(a + "\n")

print("[OK] wrote:", OUT_TXT, "n=", len(assemblies))

# 在终端，用 datasets 下载这 50 个 assembly 的蛋白
# cd /Users/xinyue/Desktop/shen/glucose

# 然后执行：
#while read acc; do
# echo "Downloading $acc"
# datasets download genome accession "$acc" \
#  --include protein \
#    --filename "data/genomes/${acc}.zip"
#done < data/glucose_assemblies_subset_50.txt

#解压并统一收集蛋白 FASTA
#for z in data/genomes/*.zip; do
#  b=$(basename "$z" .zip)
#  mkdir -p "data/genomes/$b"
#  unzip -oq "$z" -d "data/genomes/$b"
#  find "data/genomes/$b" -name "*protein.faa" -maxdepth 6 \
#    -exec cp {} "data/proteins/${b}.faa" \;
#done
