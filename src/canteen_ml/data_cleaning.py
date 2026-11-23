
import pandas as pd


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows without modifying the input."""
    return df.drop_duplicates().copy()


def replace_sentinel_values(
    df: pd.DataFrame,
    column: str = "exam_days_left",
    sentinel: int = 9999,
) -> pd.DataFrame:
    """Replace impossible sentinel values with missing values."""
    cleaned = df.copy()
    cleaned[column] = cleaned[column].replace(sentinel, float("nan"))
    return cleaned


def normalize_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize inconsistent day_type spellings."""
    cleaned = df.copy()

    aliases = {
        "wee.": "weekday",
        "hol.": "holiday",
        "sat.": "saturday",
    }

    cleaned["day_type"] = (
        cleaned["day_type"]
        .str.strip()
        .str.lower()
        .replace(aliases)
    )

    valid_categories = {"weekday", "holiday", "saturday"}
    actual_categories = set(cleaned["day_type"].dropna().unique())

    unknown = actual_categories - valid_categories

    if unknown:
        raise ValueError(f"Unknown day_type categories: {sorted(unknown)}")

    return cleaned


def remove_target_typos(
    df: pd.DataFrame,
    target: str = "plates_sold",
    multiplier: float = 10.0,
) -> pd.DataFrame:
    """
    Exclude suspicious target values exceeding a multiple
    of the target median.

    This is a training-data treatment, not a correction
    of the original target values.
    """
    cleaned = df.copy()

    median_value = cleaned[target].median()
    threshold = median_value * multiplier

    return cleaned.loc[cleaned[target] <= threshold].copy()


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the confirmed cleaning operations in sequence."""
    cleaned = remove_exact_duplicates(df)
    cleaned = replace_sentinel_values(cleaned)
    cleaned = normalize_day_type(cleaned)
    cleaned = remove_target_typos(cleaned)

    return cleaned
