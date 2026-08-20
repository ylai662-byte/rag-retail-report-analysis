from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.csv"

SCORE_COLUMNS = [
    "Avg Relevance",
    "Avg Completeness",
    "Avg Grounding",
]


def load_results():
    """Load RAG evaluation results from CSV."""
    return pd.read_csv(RESULTS_PATH)


def validate_scores(df):
    """Recalculate overall scores and verify stored results."""
    df = df.copy()

    df["Calculated Overall Score"] = (
        df[SCORE_COLUMNS].sum(axis=1).round(2)
    )

    df["Score Match"] = (
        df["Calculated Overall Score"].round(2)
        == df["Overall Score"].round(2)
    )

    return df


def rank_configurations(df):
    """Rank configurations by answer quality."""
    return df.sort_values(
        ["Calculated Overall Score", "Relevant Chunk Rate (%)"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_results(df):
    """Print evaluation summary and configuration ranking."""
    ranked = rank_configurations(df)

    best = ranked.iloc[0]
    best_retrieval = df.loc[df["Relevant Chunk Rate (%)"].idxmax()]

    print("RAG Evaluation Summary")
    print("=" * 50)

    print(f"Configurations evaluated: {len(df)}")
    print()

    print("Best answer-quality configuration")
    print("-" * 50)
    print(f"Configuration: {best['Configuration']}")
    print(f"Overall score: {best['Calculated Overall Score']:.1f}")
    print(
        f"Relevant Chunk Rate: "
        f"{best['Relevant Chunk Rate (%)']:.1f}%"
    )

    print()

    print("Highest retrieval-precision configuration")
    print("-" * 50)
    print(f"Configuration: {best_retrieval['Configuration']}")
    print(
        f"Relevant Chunk Rate: "
        f"{best_retrieval['Relevant Chunk Rate (%)']:.1f}%"
    )
    print(
        f"Overall score: "
        f"{best_retrieval['Calculated Overall Score']:.1f}"
    )

    print()

    print("Configuration ranking")
    print("-" * 50)

    display_columns = [
        "Configuration",
        "Calculated Overall Score",
        "Relevant Chunk Rate (%)",
    ]

    print(
        ranked[display_columns]
        .to_string(index=False)
    )

    print()

    if df["Score Match"].all():
        print("Score validation: PASS")
    else:
        print("Score validation: FAIL")
        print(
            df.loc[
                ~df["Score Match"],
                [
                    "Configuration",
                    "Overall Score",
                    "Calculated Overall Score",
                ],
            ]
        )


def main():
    results = load_results()
    validated_results = validate_scores(results)
    summarize_results(validated_results)


if __name__ == "__main__":
    main()