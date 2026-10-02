import pandas as pd

# Load datasets
ratings = pd.read_csv(r'C:\PentahoProjects\ml-latest-small\ratings.csv')
movies = pd.read_csv(r'C:\PentahoProjects\ml-latest-small\movies.csv')

# Group & Aggregate
agg_ratings = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    total_ratings=('rating', 'count')
).reset_index()

# Merge
df = pd.merge(movies, agg_ratings, on='movieId', how='inner')
df['recommendation_score'] = df['avg_rating'] * df['total_ratings']

# Drop raw text 'title' which breaks CSV parsing in WEKA
df = df.drop(columns=['title'])

# Clean genres string column (remove spaces/special characters)
df['genres'] = df['genres'].str.replace(' ', '_').str.replace('|', '_')

# Write directly into WEKA ARFF format
arff_path = r'C:\PentahoProjects\ml-latest-small\weka_ready_movies.arff'

with open(arff_path, 'w', encoding='utf-8') as f:
    f.write("@relation movie_recommendations\n\n")
    f.write("@attribute movieId numeric\n")
    f.write("@attribute genres string\n")
    f.write("@attribute avg_rating numeric\n")
    f.write("@attribute total_ratings numeric\n")
    f.write("@attribute recommendation_score numeric\n\n")
    f.write("@data\n")
    for _, row in df.iterrows():
        f.write(f"{int(row['movieId'])},\"{row['genres']}\",{row['avg_rating']:.4f},{int(row['total_ratings'])},{row['recommendation_score']:.4f}\n")

print("ARFF file successfully generated!")