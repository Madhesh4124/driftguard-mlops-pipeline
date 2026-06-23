import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df=pd.read_csv(r"data\UCI_Credit_Card.csv")

#Split the dataset for 2 phases; for training and kafka streaming
phase1=df.iloc[:15000].copy()
phase2=df.iloc[15000:].copy()


# Modify Phase 2 to simulate drift
phase2["AGE"] += 10
phase2["BILL_AMT1"] *= 1.7
phase2["LIMIT_BAL"] *= 0.75

#Concat the dataset
combined = pd.concat(
    [phase1, phase2],
    ignore_index=True
)

combined.to_csv(
    "data/credit_default_two_phase.csv",
    index=False
)




def plot_feature_drift(phase1, phase2, column):
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    output_dir = BASE_DIR / "reports"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(8, 5))

    # 2. Plot simple histograms using standard bins
    plt.hist(phase1[column], bins=30, alpha=0.5, label="Phase1")
    plt.hist(phase2[column], bins=30, alpha=0.5, label="Phase2")

    # 3. Add titles and legends
    plt.title(f"{column} Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Save to target Windows path with zero edge-clipping
    file_path = output_dir / f"{column}_drift.png"
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    plt.close()

# Executing the plots
plot_feature_drift(phase1, phase2, "AGE")
plot_feature_drift(phase1, phase2, "BILL_AMT1")
plot_feature_drift(phase1, phase2, "LIMIT_BAL")


#Statistics

print("\nAGE")
print(phase1["AGE"].describe())
print(phase2["AGE"].describe())

print("\nLIMIT_BAL")
print(phase1["LIMIT_BAL"].describe())
print(phase2["LIMIT_BAL"].describe())

print("\nBILL_AMT1")
print(phase1["BILL_AMT1"].describe())
print(phase2["BILL_AMT1"].describe())
