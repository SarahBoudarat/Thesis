import pandas as pd


ratings = pd.read_csv("C:/Users/sarah.boudarat/PycharmProjects/Thesis/ratings.dat",
                      sep="::",
                      engine="python",
                      names=["userId", "movieId", "rating", "timestamp"])

# some stats
num_users = ratings['userId'].nunique()
num_items = ratings['movieId'].nunique()
num_ratings = len(ratings)
rating_scale = (ratings['rating'].min(), ratings['rating'].max())
density = num_ratings / (num_users * num_items)

# user by rating volume
ratings_per_user = ratings['userId'].value_counts()
percent_100 = (ratings_per_user[ratings_per_user >= 100].count() / num_users) * 100
percent_150 = (ratings_per_user[ratings_per_user >= 150].count() / num_users) * 100
percent_200 = (ratings_per_user[ratings_per_user >= 200].count() / num_users) * 100

# print summary
print(f"Total Users: {num_users}")
print(f"Total Items: {num_items}")
print(f"Total Ratings: {num_ratings}")
print(f"Rating Scale: {rating_scale}")
print(f"Density: {density:.4f}")
print(f"Users with ≥100 ratings: {percent_100:.2f}%")
print(f"Users with ≥150 ratings: {percent_150:.2f}%")
print(f"Users with ≥200 ratings: {percent_200:.2f}%")
