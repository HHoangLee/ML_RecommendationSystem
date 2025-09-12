import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

seen_movie = "Quick and the Dead, The (1995)"
top_k = 20

# Load movie data
data = pd.read_csv(
    "data/movies.csv",
    sep="\t",
    encoding="latin1",
    usecols=["movie_id", "title", "genres"]
)

# Clean genres column
data["genres"] = data["genres"].apply(lambda s: s.replace("|", " ").replace("-", ""))

# Convert genres into TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])

# Compute pairwise cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Build similarity DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=data["title"], columns=data["title"])

# Get top-k most similar movies
top_movies = cosine_sim_df.loc[seen_movie, :].sort_values(ascending=False)[:top_k]
print(top_movies)
