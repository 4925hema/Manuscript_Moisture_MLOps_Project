"""Materialize Version 1 and Version 2 engineered features into SQLite."""
from .config import V1_PATH, V2_PATH, FEATURE_STORE_PATH
from .feature_store import materialize_features, get_features


def main() -> None:
    count = materialize_features([V1_PATH, V2_PATH], FEATURE_STORE_PATH)
    stored = get_features(FEATURE_STORE_PATH)
    print(f"Materialized {count} observations into {FEATURE_STORE_PATH}")
    print(f"Feature store rows: {len(stored)}")
    print("Stored engineered features:", "dew_point_c, dew_point_margin_c, condensation_risk_index, ventilation_effectiveness_ratio")


if __name__ == "__main__":
    main()
