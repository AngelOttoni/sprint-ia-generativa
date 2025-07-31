import pandas as pd
from pathlib import Path

# Path to the CSV file containing the books dataset
BOOKS_DATASET_PATH = Path(__file__).parent / './data/GoodReads_100k_books.csv'

def load_books_dataset() -> pd.DataFrame:
    """
    Loads the books dataset with proper error handling.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        ValueError: If there are encoding or parsing errors.
    """
    try:
        return pd.read_csv(BOOKS_DATASET_PATH, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: {BOOKS_DATASET_PATH}")
    except UnicodeDecodeError:
        raise ValueError(f"Encoding error while reading {BOOKS_DATASET_PATH}. Try latin-1 encoding.")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse {BOOKS_DATASET_PATH}. The file may be corrupted.")
