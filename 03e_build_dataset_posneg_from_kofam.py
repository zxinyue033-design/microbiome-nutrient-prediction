import os
import glob
import pandas as pd

PROJECT_DIR = "/Users/xinyue/Desktop/shen/glucose"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

POS_KOFAM_DIR = os.path.join(DATA_DIR, "ko_calls")
NEG_KOFAM_DIR = os.path.join(DATA_DIR, "ko_calls_neg")

MAP_PATH = os.path.join(DATA_DIR, "glucose_species_to_assembly.csv")
Y_PATH = os.path.join(DATA_DIR, "species_level_y.csv")

OUT_PATH = os.path.join(DATA_DIR, "dataset_glucose_KO_ml_posneg.csv")

# glucose-related KOs
GLUCOSE_KOS = [
    "K00844", "K12407", "K01810", "K00850", "K01623",
    "K01803", "K00927", "K01834", "K01689",
    "K02025", "K02026", "K02027"
]


def parse_kofam(tsv_path):
    """
    读取 *.kofam.tsv，返回所有高置信命中的 KO（带 *）
    """
    hits = set()
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("*"):
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2].startswith("K"):
                    hits.add(parts[2])
    return hits


def build_X_from_dir(kofam_dir):
    rows = []
    files = glob.glob(os.path.join(kofam_dir, "*.kofam.tsv"))

    for fp in files:
        assembly = os.path.basename(fp).replace(".kofam.tsv", "")
        kos = parse_kofam(fp)

        row = {"assembly_accession": assembly}
        for ko in GLUCOSE_KOS:
            row[ko] = int(ko in kos)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("[INFO] building X from positive and negative kofam outputs")

    X_pos = build_X_from_dir(POS_KOFAM_DIR)
    X_neg = build_X_from_dir(NEG_KOFAM_DIR)

    X_all = pd.concat([X_pos, X_neg], ignore_index=True)

    # assembly -> species
    m = pd.read_csv(MAP_PATH)
    assembly_col = "assembly_accession" if "assembly_accession" in m.columns else "assembly"
    m = m[["species", assembly_col]].rename(columns={assembly_col: "assembly_accession"})

    df = m.merge(X_all, on="assembly_accession", how="inner")

    # species-level aggregation
    X_species = df.groupby("species")[GLUCOSE_KOS].max().reset_index()

    # labels
    y = pd.read_csv(Y_PATH)
    label_col = "species_label" if "species_label" in y.columns else "label"

    dataset = y[["species", label_col]] \
        .merge(X_species, on="species", how="inner") \
        .rename(columns={label_col: "species_label"})

    dataset.to_csv(OUT_PATH, index=False)

    print(f"[OK] saved dataset: {OUT_PATH}")
    print("\nLabel distribution:")
    print(dataset["species_label"].value_counts())
    print("\nFeature counts:")
    print(dataset[GLUCOSE_KOS].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()