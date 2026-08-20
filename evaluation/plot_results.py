import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "results.csv")

df = df.sort_values("Overall Score", ascending=True)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    df["Configuration"],
    df["Overall Score"]
)

plt.xlabel("Overall Answer Quality Score")
plt.ylabel("RAG Configuration")
plt.title("RAG Configuration Evaluation")

plt.xlim(0, 10)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}",
        va="center"
    )

plt.tight_layout()

output_path = BASE_DIR / "rag_evaluation.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Saved chart to: {output_path}")