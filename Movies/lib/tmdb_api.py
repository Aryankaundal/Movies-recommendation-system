import os
import requests
import streamlit as st
from typing import List, Dict
import pandas as pd
import numpy as np

# Free TMDb API key (demo - replace with your own from themoviedb.org)
DEFAULT_API_KEY = YOUR_TMDB_API_KEY


class TMDbAPI:
    def __init__(self, api_key: str = None):
        """Initialize TMDb API client"""
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def get_popular_movies(self, pages: int = 10) -> pd.DataFrame:
        movies = []
        
        for page in range(1, pages + 1):
            data = self._make_request("movie/popular", {"page": page})
            
            for movie in data.get('results', []):
                movies.append({
                    'movie_id': movie['id'],
                    'title': movie['title'],
                    'year': int(movie.get('release_date', '2000-01-01')[:4]) if movie.get('release_date') else 2000,
                    'genre': self._get_genre_name(movie.get('genre_ids', [])[0]) if movie.get('genre_ids') else 'Unknown',
                    'rating': round(movie.get('vote_average', 0), 1),
                    'popularity': round(movie.get('popularity', 0), 1),
                    'vote_count': movie.get('vote_count', 0),
                    'overview': movie.get('overview', ''),
                    'poster_path': self.get_poster_url(movie.get('poster_path', '')),
                    'director': 'Various',  # Would need extra API call
                    'duration': 120  # Default duration
                })
        
        return pd.DataFrame(movies)
    
    def get_top_rated_movies(self, pages: int = 10) -> pd.DataFrame:
        movies = []
        
        for page in range(1, pages + 1):
            data = self._make_request("movie/top_rated", {"page": page})
            
            for movie in data.get('results', []):
                movies.append({
                    'movie_id': movie['id'],
                    'title': movie['title'],
                    'year': int(movie.get('release_date', '2000-01-01')[:4]) if movie.get('release_date') else 2000,
                    'genre': self._get_genre_name(movie.get('genre_ids', [])[0]) if movie.get('genre_ids') else 'Unknown',
                    'rating': round(movie.get('vote_average', 0), 1),
                    'popularity': round(movie.get('popularity', 0), 1),
                    'vote_count': movie.get('vote_count', 0),
                    'overview': movie.get('overview', ''),
                    'poster_path': self.get_poster_url(movie.get('poster_path', '')),
                    'director': 'Various',
                    'duration': 120
                })
        
        return pd.DataFrame(movies)
    
    def search_movies(self, query: str) -> pd.DataFrame:
        data = self._make_request("search/movie", {"query": query})
        movies = []
        
        for movie in data.get('results', []):
            movies.append({
                'movie_id': movie['id'],
                'title': movie['title'],
                'year': int(movie.get('release_date', '2000-01-01')[:4]) if movie.get('release_date') else 2000,
                'genre': self._get_genre_name(movie.get('genre_ids', [])[0]) if movie.get('genre_ids') else 'Unknown',
                'rating': round(movie.get('vote_average', 0), 1),
                'popularity': round(movie.get('popularity', 0), 1),
                'vote_count': movie.get('vote_count', 0),
                'overview': movie.get('overview', ''),
                'poster_path': self.get_poster_url(movie.get('poster_path', '')),
                'director': 'Various',
                'duration': 120
            })
        
        return pd.DataFrame(movies)
    
    def _get_genre_name(self, genre_id: int) -> str:
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi', 10770: 'TV Movie',
            53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return genre_map.get(genre_id, 'Unknown')
    
    def generate_simulated_ratings(self, movies_df: pd.DataFrame, num_users: int = 100) -> pd.DataFrame:
        ratings = []
        
        for user_id in range(1, num_users + 1):
            # Each user rates 10-30 random movies
            num_ratings = np.random.randint(10, 31)
            rated_movies = movies_df.sample(n=num_ratings)
            
            for _, movie in rated_movies.iterrows():
                # Generate rating influenced by movie rating
                base_rating = movie['rating'] / 2  # Convert 10-scale to 5-scale
                rating = np.clip(
                    np.random.normal(base_rating, 0.8),
                    1, 5
                )
                
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie['movie_id'],
                    'rating': round(rating, 1)
                })
        
        return pd.DataFrame(ratings)
    
    def get_poster_url(self, poster_path: str) -> str:
        if poster_path:
            return f"{self.image_base_url}{poster_path}"
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_movies_from_api(api_key: str = None, pages: int = 25):
    api = TMDbAPI(api_key)
    
    with st.spinner("Fetching movies from TMDb API..."):
        # Fetch both popular and top rated
        popular_movies = api.get_popular_movies(pages=pages // 2)
        top_rated_movies = api.get_top_rated_movies(pages=pages // 2)
        
        # Combine and remove duplicates
        movies_df = pd.concat([popular_movies, top_rated_movies]).drop_duplicates(subset=['movie_id'])
        
        # Filter to movies from 1980 onwards
        movies_df = movies_df[movies_df['year'] >= 1980].reset_index(drop=True)
        
        # Generate simulated ratings
        ratings_df = api.generate_simulated_ratings(movies_df, num_users=100)
    
    return movies_df, ratings_df
