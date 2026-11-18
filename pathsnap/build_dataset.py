"""Build the shared processed dataset.  Owner: Zubed.  (Brief 6.3 / 6.4)

Run from repo root:  python -m pathsnap.build_dataset
Deterministic (seed 42), so anyone can regenerate identical files instead of waiting for a copy.
Writes data/processed/{train,test}_candidates.parquet, point_refs.parquet and folds.csv
(folds.csv is committed to git; the parquet files are shared via the team Drive or regenerated).
"""


def main():
    raise NotImplementedError("Zubed: implement (brief 6.3)")


if __name__ == "__main__":
    main()
