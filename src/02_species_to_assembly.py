import os
import time
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
from config import DATA_DIR 

IN_Y = os.path.join(DATA_DIR, "y_glu.csv")
OUT_MAP = os.path.join(DATA_DIR, "glucose_species_to_assembly.csv")

# Load y table & species list
y_glucose = pd.read_csv(IN_Y)

species_list = (
    y_glucose["species"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print("Unique species:", len(species_list))

# NCBI helper functions

# 1 species → taxid
def species_to_taxid(species_name):
    """species scientific name -> NCBI taxid"""
    try:
        h = Entrez.esearch(
            db="taxonomy",
            term=f"{species_name}[Scientific Name]",
            retmax=1
        )
        r = Entrez.read(h)
        h.close()

        if not r["IdList"]:
            return None
        return r["IdList"][0]

    except Exception:
        return None

# 2 taxid → best assembly (RefSeq preferred)
def taxid_to_best_assembly(taxid):
    """
    taxid -> assembly accession
    ① RefSeq (GCF) 优先
    ② 若无 RefSeq -> fallback 到 GenBank (GCA)
    """
    try:
        # --- RefSeq first ---
        h = Entrez.esearch(
            db="assembly",
            term=f"txid{taxid}[Organism:exp] AND refseq[filter]",
            retmax=1
        )
        r = Entrez.read(h)
        h.close()

        if r["IdList"]:
            asm_id = r["IdList"][0]
        else:
            # --- fallback to all assemblies ---
            h = Entrez.esearch(
                db="assembly",
                term=f"txid{taxid}[Organism:exp]",
                retmax=1
            )
            r = Entrez.read(h)
            h.close()

            if not r["IdList"]:
                return None
            asm_id = r["IdList"][0]

        h = Entrez.esummary(db="assembly", id=asm_id, report="full")
        s = Entrez.read(h)
        h.close()

        doc = s["DocumentSummarySet"]["DocumentSummary"][0]
        return doc.get("AssemblyAccession")

    except Exception:
        return None

# Main loop: species → taxid → assembly
rows = []

for sp in tqdm(species_list, desc="Mapping species"):
    taxid = species_to_taxid(sp)
    acc = taxid_to_best_assembly(taxid) if taxid else None

    rows.append({
        "species": sp,
        "taxid": taxid,
        "assembly_accession": acc
    })

    time.sleep(0.3)  # 防止 NCBI 限流

# 5. Save & summary
df_map = pd.DataFrame(rows)
df_map.to_csv(OUT_MAP, index=False)

print("Saved mapping:", OUT_MAP)
print("Mapping coverage (non-null assembly):")
print(df_map["assembly_accession"].notna().value_counts())

print(df_map.head())
# endregion

import re
import urllib.request
from pathlib import Path

IN_MAP = Path(DATA_DIR) / "glucose_species_to_assembly.csv"
GENOME_DIR = Path(DATA_DIR) / "glucose_genomes_fna"
GENOME_DIR.mkdir(parents=True, exist_ok=True)

df_map = pd.read_csv(IN_MAP)
df_ok = df_map.dropna(subset=["assembly_accession"]).copy()
print("Total:", len(df_map), "With assembly:", len(df_ok))

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120]

def get_ftp_path_for_accession(assembly_acc: str):
    # 用 accession 查 assembly record -> FTP 路径
    h = Entrez.esearch(db="assembly", term=f"{assembly_acc}[Assembly Accession]", retmax=1)
    r = Entrez.read(h); h.close()
    if not r["IdList"]:
        return None

    asm_id = r["IdList"][0]
    h = Entrez.esummary(db="assembly", id=asm_id, report="full")
    s = Entrez.read(h); h.close()
    doc = s["DocumentSummarySet"]["DocumentSummary"][0]

    # RefSeq 优先，否则 GenBank
    return doc.get("FtpPath_RefSeq") or doc.get("FtpPath_GenBank")

def download_genomic_fna_gz(ftp_path: str, save_path: Path):
    base = ftp_path.rstrip("/").split("/")[-1]
    url = f"{ftp_path}/{base}_genomic.fna.gz"
    urllib.request.urlretrieve(url, save_path)
    return url

success, failed = [], []

for _, row in tqdm(df_ok.iterrows(), total=len(df_ok)):
    sp = row["species"]
    acc = row["assembly_accession"]

    out_file = GENOME_DIR / f"{safe_name(sp)}__{acc}_genomic.fna.gz"
    if out_file.exists():
        success.append((sp, acc, "exists"))
        continue

    ftp = get_ftp_path_for_accession(acc)
    if not ftp:
        failed.append((sp, acc, "no_ftp"))
        continue

    try:
        url = download_genomic_fna_gz(ftp, out_file)
        success.append((sp, acc, url))
    except Exception as e:
        failed.append((sp, acc, str(e)))

    time.sleep(0.3)

print("\nDownloaded:", len(success))
print("Failed:", len(failed))
print("Genome folder:", str(GENOME_DIR))
failed[:5]