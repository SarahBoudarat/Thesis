import pandas as pd

import pandas as pd

# Load data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Count unique users and unique items
num_users = ratings["userId"].nunique()
num_items = movies["movieId"].nunique()

print(f"Number of unique users: {num_users}")
print(f"Number of unique items: {num_items}")

# Load ratings.dat and movies.dat using the correct encoding and delimiter
ratings_1m = pd.read_csv(
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\ratings.dat",
    sep="::", engine="python", encoding="latin-1",
    names=["userId", "movieId", "rating", "timestamp"]
)

movies_1m = pd.read_csv(
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\movies.dat",
    sep="::", engine="python", encoding="latin-1",
    names=["movieId", "title", "genres"]
)

# Count unique users and movies
num_users = ratings_1m["userId"].nunique()
num_items = movies_1m["movieId"].nunique()

print(f"Number of unique users (MovieLens 1M): {num_users}")
print(f"Number of unique movies (MovieLens 1M): {num_items}")
rated_movies_100k = ratings_100k["movieId"].nunique()
rated_movies_1m = ratings_1m["movieId"].nunique()

print(f"Rated items in 100K: {rated_movies_100k}")
print(f"Rated items in 1M: {rated_movies_1m}")