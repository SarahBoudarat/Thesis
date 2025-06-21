import pandas as pd

# Load the data
ratings = pd.read_csv("ratings.csv")   # path to MovieLens 100K ratings file
movies = pd.read_csv("movies.csv")     # path to MovieLens 100K movies file

# Compute statistics
num_users = ratings['userId'].nunique()
num_items = ratings['movieId'].nunique()
num_ratings = len(ratings)
rating_scale = "[1-5]"
density = (num_ratings / (num_users * num_items)) * 100

# Display
print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Ratings: {num_ratings}")
print(f"Rating Scale: {rating_scale}")
print(f"Density: {density:.3f}%")

import pandas as pd

ratings = pd.read_csv("ratings.csv")
user_counts = ratings.groupby("userId").size()
active_users = user_counts[user_counts >= 50]
percentage = len(active_users) / ratings["userId"].nunique() * 100
print(f"{percentage:.2f}%")
