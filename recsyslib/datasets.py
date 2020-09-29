import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from tensorflow import keras
import itertools


DATAFOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NUM_RATINGS = "25m"
FILENAME = f"ml-{NUM_RATINGS}.zip"


class MovieLensData:
    def __init__(
        self,
        data_folder=DATAFOLDER,
        num_ratings=NUM_RATINGS,
        filename=FILENAME,
    ):
        self.data_folder = data_folder
        self.num_ratings = num_ratings
        self.filename = filename
        df = self.get_ml_data()
        self.df = self._assign_indices(df)

    def get_ml_data(self):

        local_datadir = os.path.join(
            self.data_folder, self.filename.replace(".zip", "")
        )

        movielens_data_file_url = os.path.join(
            "http://files.grouplens.org/datasets/movielens", self.filename
        )

        movielens_zipped_file = keras.utils.get_file(
            self.filename, movielens_data_file_url, extract=False
        )

        # Only extract the data the first time the script is run.
        if not os.path.exists(local_datadir):
            print(f"retrieving dataset from {movielens_data_file_url}")
            with ZipFile(movielens_zipped_file, "r") as zip:
                zip.extractall(path=local_datadir)

        ratings_file = os.path.join(local_datadir, "ml-25m/ratings.csv")
        movies_file = os.path.join(local_datadir, "ml-25m/movies.csv")
        return pd.read_csv(ratings_file).merge(
            pd.read_csv(movies_file), on="movieId"
        )

    def get_item_sequences(self):
        self.df["movieIdstr"] = self.df["movieId"].astype(str)
        seq = (
            self.df.sort_values(["userId", "timestamp"])
            .groupby(["userId"])["movieIdstr"]
            .apply(list)
        )
        return seq

    def _assign_indices(self, df):
        # assign monotonically increasing ids (matrix/nn search indices) to users and movies
        users = df["userId"].unique()
        movies = df["movieId"].unique()
        self.num_users = len(users)
        self.num_items = len(movies)

        # userIds -> userIndices
        userId_to_idx = {u: i for i, u in enumerate(users)}
        movieId_to_idx = {m: i for i, m in enumerate(movies)}

        df["user_idx"] = df["userId"].map(userId_to_idx)
        df["movie_idx"] = df["movieId"].map(movieId_to_idx)

        # dataframe is sorted by user, timestamp
        # break correlations
        df = df.sample(frac=1, random_state=49)
        return df

    def downsample_data(self, num_users=100, num_items=100):
        users = (
            self.df.groupby("userId")
            .size()
            .reset_index(name="counts")
            .sort_values(["counts"], ascending=False)
            .head(num_users)
        )["userId"].tolist()
        self.df = self.df[self.df["userId"].isin(users)]

        movies = (
            self.df.groupby("movieId")
            .size()
            .reset_index(name="counts")
            .sort_values(["counts"], ascending=False)
            .head(num_items)
        )["movieId"].tolist()

        if len(movies) < num_items:
            print(f"using on {len(movies)} for this number of users")
            num_items = len(movies)

        self.df = self.df[self.df["movieId"].isin(movies)]
        self.num_users = num_users
        self.num_items = num_items
        self.df = self._assign_indices(self.df)

    def _get_map(self, key, val):
        return {
            k: v
            for k, v in self.df[[key, val]]
            .drop_duplicates()
            .itertuples(index=False)
        }

    def get_user_idx_map(self):
        return self._get_map("userId", "user_idx")

    def get_item_idx_map(self):
        return self._get_map("movieId", "movie_idx")

    def get_item_idx_to_title(self):
        return self._get_map("movie_idx", "title")

    @staticmethod
    def scale_rating(values):
        # scale rating [0 - 1]
        minval = np.min(values)
        maxval = np.max(values)
        return (values - minval) / (maxval - minval)

    def get_user_item_rating_tuples(self, scale=False, split=0.9):
        X = (self.df["user_idx"].values, self.df["movie_idx"].values)
        y = self.df["rating"].values
        if scale:
            y = self.scale_rating(y)
        if not split:
            return X, y
        else:
            training = int(self.df.shape[0] * split)
            X_train, X_test = X[:training], X[training:]
            y_train, y_test = y[:training], y[training:]
            return (X_train, X_test, y_train, y_test)

    def get_genre_movie_data(self):
        movies = self.df[
            ["movieId", "movie_idx", "title", "genres"]
        ].drop_duplicates()
        movies["genres"] = movies["genres"].str.split("|")
        max_movie_idx = np.max(movies["movie_idx"].values)

        all_genres = set(itertools.chain.from_iterable(movies["genres"]))
        genre_map = {
            g: i + max_movie_idx for i, g in enumerate(list(all_genres), 1)
        }

        def other_labels(g, all_genre=all_genres, gmap=genre_map):
            return [genre_map[g] for g in list(all_genres.difference(set(g)))]

        movies["negative_genre_idx"] = movies["genres"].apply(other_labels)
        movies = movies.explode("genres")
        movies["genre_idx"] = movies["genres"].map(genre_map)

        def sample_negatives(negative_genre_idx):
            n_neg_genres = min(len(negative_genre_idx), 10)
            return np.random.choice(negative_genre_idx, n_neg_genres)

        movies["negative_genre_idx"] = movies["negative_genre_idx"].apply(
            sample_negatives
        )
        movies["year"] = movies["title"].str.extract("([0-9]{4})")
        return movies
