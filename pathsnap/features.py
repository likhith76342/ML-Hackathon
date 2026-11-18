"""Feature definitions.  Owner: Zubed.  (Brief 6.3 step 4)

FEATURE_COLUMNS is the single source of truth. Every model trains on exactly these columns
and the API selects exactly these columns. Features must never use .route or labels.
"""
FEATURE_COLUMNS: list[str] = [
    # Zubed: finalise this list and announce it. Suggested start (brief 6.3):
    # "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "bearing_diff",
    # "speed_mps", "heading_valid", "dt_prev", "dt_next", "n_candidates",
    # "arc_length_m", "proj_t",
]
