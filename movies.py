# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:01:05 2023

@author: Laptop
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# Load the MovieLens dataset
df = pd.read_csv('D:/DOWNLOADS/movies.csv')

# Pre-process the dataset
df.dropna(inplace=True)
df['genres'] = df['genres'].str.split('|')
for i in range(len(df)):
    df.loc[i, 'genres'] = ','.join(df.loc[i, 'genres'])

# Convert the genres column to a regular column
df.loc[:, 'genres'] = df.loc[:, 'genres'].apply(lambda x: str(x))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['movieId', 'title']], df['genres'].tolist(), test_size=0.2)

# Implement a collaborative filtering algorithm
model = NearestNeighbors(n_neighbors=10)
model.fit(X_train)

# Train the model using the training set
model.fit(X_train)

# Evaluate the model's performance on the testing set
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions)
print('MAE:', mae)
print('RMSE:', rmse)

# Use the model to make movie recommendations for users based on their preferences
def get_recommendations(user_id, num_recommendations):
    distances, indices = model.kneighbors(X_test.loc[user_id, :])
    recommended_movies = X_test.iloc[indices[0]].sort_values('rating.csv', ascending=False)[:num_recommendations]
    return recommended_movies

# Test the model by inputting new user ratings
new_user_ratings = {
    1: [4, 5, 3, 1, 2],
    2: [5, 5, 4, 4, 5]
}
for user_id, ratings in new_user_ratings.items():
    recommendations = get_recommendations(user_id, 5)
    print('User {}:'.format(user_id))
    for movie in recommendations:
        print('   ', movie[0], movie[1])