import os
import pandas as pd
from config import DATA_DIR

IN_CSV = os.path.join(DATA_DIR, "y_glucose.csv")
OUT_Y  = os.path.join(DATA_DIR, "y_glu.csv")


def build_y_glucose():
    df = pd.read_csv(IN_CSV, encoding="latin1")

    df["ID"] = df["ID"].ffill()
    df["species"] = df["species"].ffill()

    glucose_names = {"glucose", "D-glucose", "alpha-D-glucose"}
    df_glu = df[df["Metabolite (utilization)"].isin(glucose_names)].copy()
    df_glu = df_glu[df_glu["Utilization activity"].isin(["+", "-"])].copy()

    df_glu["label"] = df_glu["Utilization activity"].map({"+": 1, "-": 0})
    df_glu["metabolite"] = "glucose"

    y_glucose = df_glu[
        ["ID", "species", "metabolite", "label", "Kind of utilization tested"]
    ].rename(columns={
        "ID": "bacdive_id",
        "Kind of utilization tested": "test_type"
    }).drop_duplicates()

    y_glucose.to_csv(OUT_Y, index=False)
    print("Saved:", OUT_Y, y_glucose.shape)


if __name__ == "__main__":
    build_y_glucose()


IN_Y = os.path.join(DATA_DIR, "y_glu.csv")
OUT_Y = os.path.join(DATA_DIR, "species_level_y.csv")


# Load strain-level labels
y = pd.read_csv(IN_Y)

# 确保列存在
required_cols = {"species", "label"}
missing = required_cols - set(y.columns)
if missing:
    raise ValueError(f"Missing columns in y_glucose.csv: {missing}")

# 聚合到 species level
# 对每个 species，收集它所有 label
agg = (
    y.groupby("species")["label"]
    .apply(lambda s: set(s.dropna().astype(int)))
    .reset_index(name="label_set")
)

# 判断是否冲突：同时出现 0 和 1
agg["is_conflict"] = agg["label_set"].apply(
    lambda s: (0 in s and 1 in s)
)

# 只保留“无冲突”的 species
species_y = agg[~agg["is_conflict"]].copy()

# 生成最终 species_label
species_y["species_label"] = species_y["label_set"].apply(
    lambda s: 1 if 1 in s else 0
)

# 只保留需要的列
species_y = species_y[["species", "species_label"]].reset_index(drop=True)


species_y.to_csv(OUT_Y, index=False)

print(f"[OK] species_level_y saved to: {OUT_Y}")
print("Number of species (no conflict):", len(species_y))
print(species_y.head())
