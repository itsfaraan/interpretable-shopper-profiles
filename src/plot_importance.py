import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("reports/surrogate_feature_importance.csv")


df = df.sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=df, color="#2ca02c") 

plt.title("What Drives the Personas? (Global Feature Importance)", fontsize=16, fontweight="bold")
plt.xlabel("Relative Importance (Contribution to Decision Tree)", fontsize=12)
plt.ylabel("")
plt.xticks(fontsize=11)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}")) # Format as percentage

plt.tight_layout()
plt.savefig("feature_importance_slide.png", dpi=300)
print("Saved feature_importance_slide.png to your folder!")