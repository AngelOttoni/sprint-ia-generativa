import pandas as pd
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('books_search_engine')

# Path to the CSV file containing the books dataset
books_dataset = Path(__file__).parent / "./data/GoodReads_100k_books.csv"

@mcp.tool()
def search_engine(genre: str, pg_number: int, csv_path=books_dataset):
    """
    Search for books in a dataset by filtering based on genre and approximate
    page count, returning the top 10 results sorted by rating.

    Args:
        genre (str): Book genre to filter by (e.g., 'romance', 'fantasy').
        pg_number (int): Approximate desired number of pages.
        csv_path (str | Path, optional): Path to the CSV dataset file.
                                         Defaults to 'books_dataset'.

    Returns:
        list[dict]: A list of up to 10 books (as dictionaries) containing:
                    'title', 'author', 'pages', 'genre', 'rating', 'desc'.
    """
    # Load the dataset from the CSV file
    dataset = pd.read_csv(csv_path, encoding="utf-8")

    # Ensure 'pages' and 'rating' columns are in the correct numeric types
    dataset["pages"] = dataset["pages"].astype(int)
    dataset["rating"] = dataset["rating"].astype(float)

    # Filter books that match the genre (case-insensitive)
    # and have a page count within ±20 pages of the desired value
    dataset = dataset[
        (dataset["genre"].str.contains(genre, case=False))
        & ((pg_number - 20) < dataset["pages"])
        & (dataset["pages"] < (pg_number + 20))
    ]

    # If there are too many results (>100), keep only books with rating > 4
    if len(dataset) > 100:
        dataset_4 = dataset[dataset["rating"] > 4]
        if len(dataset_4) > 0:
            dataset = dataset_4

    # Sort books by rating in descending order
    dataset = dataset.sort_values(by="rating", ascending=False)

    # Select relevant columns and return the top 10 results as a list of dicts
    result = (
        dataset[["title", "author", "pages", "genre", "rating", "desc"]]
        .head(10)
        .to_dict(orient="records")
    )

    return result


if __name__ == "__main__":
    # Example usage: search for romance books with around 250 pages
    mcp.run(transport='stdio')
