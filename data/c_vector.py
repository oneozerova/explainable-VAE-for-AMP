import pandas as pd
from pathlib import Path

def save_df(out_path, df):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "Data" / "processed" / "data.csv"
anticancer_path = project_root / "Data" / "processed" / "anticancer.csv"
antiparasitic_path = project_root / "Data" / "processed" / "antiparasitic.csv"
viral_path = project_root / "Data" / "processed" / "viral.csv"

# data
data_df = pd.read_csv(data_path)
anticancer_df = pd.read_csv(anticancer_path)
antiparasitic_df = pd.read_csv(antiparasitic_path)
viral_df = pd.read_csv(viral_path)

datasets_names = [anticancer_df, antiparasitic_df, viral_df]

for dataset in datasets_names:
    text = (
            dataset["Activity"].fillna("").astype(str)
    )

    m_gram_pos = text.str.contains(r"gram\+", case=False, regex=True, na=False)
    m_gram_neg = text.str.contains(r"gram-", case=False, regex=True, na=False)

    dataset["is_anti_gram_positive"] = m_gram_pos.astype(int)
    dataset["is_anti_gram_negative"] = m_gram_neg.astype(int)
    dataset["is_antibacterial"] = (m_gram_pos | m_gram_neg).astype(int)
    dataset["is_antifungal"] = text.str.contains(r"\bantifungal\b",     case=False, regex=True, na=False).astype(int)
    dataset["is_antiviral"] = text.str.contains(r"\bantiviral\b", case=False, regex=True, na=False).astype(int)
    dataset["is_antiparasitic"] = text.str.contains(r"\bantiparasitic\b",  case=False, regex=True, na=False).astype(int)
    dataset["is_anticancer"] = text.str.contains(r"\banticancer\b",     case=False, regex=True, na=False).astype(int)

master_df = pd.concat([data_df, anticancer_df, antiparasitic_df, viral_df], ignore_index=True)
seq_col = "Sequence"
condition_cols = [
    'is_antibacterial', 'is_anti_gram_positive', 'is_anti_gram_negative',
    'is_antifungal', 'is_antiviral', 'is_antiparasitic', 'is_anticancer'
]
cond_present = [c for c in condition_cols if c in master_df.columns]
for c in cond_present:
    master_df[c] = master_df[c].fillna(0).astype(int)
agg = {c: "max" for c in cond_present}
for c in master_df.columns:
    if c not in agg and c != seq_col:
        agg[c] = "first"
master_df = (
    master_df
    .dropna(subset=[seq_col])
    .groupby(seq_col, as_index=False)
    .agg(agg)
)
print(f"Size before cleaning: {master_df.shape}")
print(master_df.head())

out_path = project_root / "Data" / "processed" / "master_dataset_before_cleaning.csv"
save_df(out_path, master_df)

# Cleaning
id_col = 'APD ID'
seq_col = 'Sequence'
len_col = 'Length'

existing_cols = [col for col in [id_col, seq_col, len_col] + condition_cols if col in master_df.columns]
df = master_df[existing_cols].dropna(subset=[seq_col])

for col in condition_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

print(f"Size after cleaning: {df.shape}")
print(df.head())

out_path = project_root / "Data" / "processed" / "master_dataset.csv"
save_df(out_path, df)

# print(df.iloc[16])