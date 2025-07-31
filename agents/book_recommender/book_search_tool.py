import pandas as pd
from mcp.server.fastmcp import FastMCP
from .utils import load_books_dataset

mcp = FastMCP('books_search_tool')

@mcp.tool()
def search_books(genre: str, pg_number: int):
    """
    Search for books in a dataset by filtering based on genre and approximate
    page count, returning the top 10 results sorted by rating.

    Args:
        genre (str): Book genre to filter by (e.g., 'romance', 'fantasy').
        pg_number (int): Approximate desired number of pages.

    Returns:
        list[dict]: A list of up to 10 books (as dictionaries) containing:
                    'title', 'author', 'pages', 'genre', 'rating', 'desc'.
    """
    # Load the dataset from the CSV file
    books_df = load_books_dataset()

    # Ensure 'pages' and 'rating' columns are in the correct numeric types
    books_df['pages'] = books_df['pages'].astype(int)
    books_df['rating'] = books_df['rating'].astype(float)

    # Filter books that match the genre (case-insensitive)
    # and have a page count within ±20 pages of the desired value
    books_df = books_df[
        (books_df['genre'].str.contains(genre, case=False))
        & ((pg_number - 20) < books_df['pages'])
        & (books_df['pages'] < (pg_number + 20))
    ]

    # If there are too many results (>100), keep only books with rating > 4
    if len(books_df) > 100:
        books_df_4 = books_df[books_df['rating'] > 4]
        if len(books_df_4) > 0:
            books_df = books_df_4

    # Sort books by rating in descending order
    books_df = books_df.sort_values(by='rating', ascending=False)

    # Select relevant columns and return the top 10 results as a list of dicts
    result = (
        books_df[['title', 'author', 'pages', 'genre', 'rating', 'desc']]
        .head(10)
        .to_dict(orient='records')
    )

    return result


@mcp.tool()
def search_book_by_title(title: str):
    """
    Search for a book by its title (case-insensitive, partial match).
    The title should be provided in English.

    Args:
        title (str): The book title to search for.

    Returns:
        list[dict]: List of books matching the title, each represented as a dictionary
                    containing keys such as author, bookformat, desc, genre, img, isbn,
                    isbn13, link, pages, rating, reviews, title, and totalratings.
    """
    books_df = load_books_dataset()
    filtered_df = books_df[books_df['title'].str.contains(title, case=False, na=False)]

    # Convert filtered DataFrame to list of dictionaries
    return filtered_df.to_dict(orient='records')


if __name__ == '__main__':
    # Run the MCP server with stdio transport for local testing
    mcp.run(transport='stdio')
