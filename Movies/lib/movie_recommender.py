import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        """Initialize the recommender with movies and ratings data"""
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.content_similarity = None
        self.knn_model = None
        self.user_movie_matrix = None
        
    def build_content_based_model(self):
        """Build content-based filtering model using movie features"""
        # Combine features: genre, director, year decade, and overview keywords
        self.movies_df['decade'] = (self.movies_df['year'] // 10) * 10
        
        # Create combined features
        features_list = []
        for _, row in self.movies_df.iterrows():
            features = f"{row['genre']} {row['director']} {row['decade']}"
            
            # Add overview keywords if available
            if 'overview' in row and pd.notna(row['overview']):
                # Take first 100 characters of overview
                overview_snippet = str(row['overview'])[:100]
                features += f" {overview_snippet}"
            
            features_list.append(features)
        
        self.movies_df['features'] = features_list
        
        # Create TF-IDF matrix with improved parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(self.movies_df['features'])
        
        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
    def build_collaborative_model(self):
        """Build collaborative filtering model using user ratings"""
        # Create user-movie matrix
        user_movie_df = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        self.user_movie_matrix = user_movie_df
        
        # Build KNN model with more neighbors
        n_neighbors = min(20, len(user_movie_df))
        self.knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=n_neighbors
        )
        self.knn_model.fit(csr_matrix(user_movie_df.values))
        
    def get_content_recommendations(self, movie_id, n=10):
        """Get recommendations based on content similarity"""
        if self.content_similarity is None:
            self.build_content_based_model()
        
        # Get movie index
        movie_matches = self.movies_df[self.movies_df['movie_id'] == movie_id]
        
        if len(movie_matches) == 0:
            return pd.DataFrame()
        
        idx = movie_matches.index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding itself)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, n=10):
        """Get recommendations based on collaborative filtering"""
        if self.knn_model is None:
            self.build_collaborative_model()
            
        # Get user index
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
            
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        
        # Find similar users (increased from 6 to 11)
        n_neighbors = min(11, len(self.user_movie_matrix))
        distances, indices = self.knn_model.kneighbors(
            self.user_movie_matrix.iloc[user_idx].values.reshape(1, -1),
            n_neighbors=n_neighbors
        )
        
        # Get movies rated highly by similar users but not rated by current user
        similar_users = indices.flatten()[1:]  # Exclude the user itself
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values
        
        # Aggregate ratings from similar users with weighted average
        recommendations_dict = {}
        recommendations_count = {}
        
        for idx, similar_user_idx in enumerate(similar_users):
            similar_user_id = self.user_movie_matrix.index[similar_user_idx]
            similarity = 1 - distances[0][idx + 1]  # Convert distance to similarity
            
            similar_user_ratings = self.ratings_df[
                (self.ratings_df['user_id'] == similar_user_id) & 
                (self.ratings_df['rating'] >= 4)
            ]
            
            for _, row in similar_user_ratings.iterrows():
                movie_id = row['movie_id']
                if movie_id not in user_ratings:
                    if movie_id not in recommendations_dict:
                        recommendations_dict[movie_id] = 0
                        recommendations_count[movie_id] = 0
                    
                    recommendations_dict[movie_id] += row['rating'] * similarity
                    recommendations_count[movie_id] += 1
        
        # Calculate weighted average
        for movie_id in recommendations_dict:
            recommendations_dict[movie_id] /= recommendations_count[movie_id]
        
        # Sort and get top N
        sorted_recs = sorted(recommendations_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        movie_ids = [x[0] for x in sorted_recs]
        
        recommendations = self.movies_df[self.movies_df['movie_id'].isin(movie_ids)].copy()
        
        # Add predicted scores in the same order
        score_dict = dict(sorted_recs)
        recommendations['predicted_score'] = recommendations['movie_id'].map(score_dict)
        recommendations = recommendations.sort_values('predicted_score', ascending=False)
        
        return recommendations
    
    def get_popular_movies(self, n=10):
        """Get most popular movies based on ratings"""
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'avg_rating', 'num_ratings']
        
        # Filter movies with at least 5 ratings (lowered from 10 for larger dataset)
        popular = movie_stats[movie_stats['num_ratings'] >= 5]
        popular = popular.sort_values('avg_rating', ascending=False).head(n)
        
        result = self.movies_df[self.movies_df['movie_id'].isin(popular['movie_id'])].merge(
            popular, on='movie_id'
        )
        
        return result.sort_values('avg_rating', ascending=False)
    
    def get_movies_by_genre(self, genre):
        """Get movies filtered by genre"""
        return self.movies_df[self.movies_df['genre'] == genre].sort_values('rating', ascending=False)
    
    def search_movies(self, query):
        """Search movies by title"""
        query = query.lower()
        results = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query, na=False)
        ]
        return results
    
    def get_trending_movies(self, n=10):
        """Get trending movies based on popularity and vote count"""
        if 'popularity' in self.movies_df.columns and 'vote_count' in self.movies_df.columns:
            # Calculate trending score
            trending = self.movies_df.copy()
            trending['trending_score'] = trending['popularity'] * np.log1p(trending['vote_count'])
            trending = trending.sort_values('trending_score', ascending=False).head(n)
            return trending
        else:
            return self.get_popular_movies(n)
