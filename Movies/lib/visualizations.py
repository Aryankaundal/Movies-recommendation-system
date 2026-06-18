"""
Visualization functions for movie recommendation system
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def setup_style():
    """Setup plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def plot_genre_distribution(movies_df):
    """Plot distribution of movies by genre"""
    setup_style()
    
    genre_counts = movies_df['genre'].value_counts()
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        labels={'x': 'Genre', 'y': 'Number of Movies'},
        title='Movie Distribution by Genre',
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig


def plot_ratings_distribution(ratings_df):
    """Plot distribution of user ratings"""
    setup_style()
    
    fig = px.histogram(
        ratings_df,
        x='rating',
        nbins=5,
        labels={'rating': 'Rating', 'count': 'Frequency'},
        title='Distribution of User Ratings',
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        height=400
    )
    
    return fig


def plot_top_rated_movies(movies_df, n=15):
    """Plot top rated movies"""
    setup_style()
    
    top_movies = movies_df.nlargest(n, 'rating')[['title', 'rating']].sort_values('rating')
    
    fig = px.bar(
        top_movies,
        x='rating',
        y='title',
        orientation='h',
        labels={'rating': 'Average Rating', 'title': 'Movie'},
        title=f'Top {n} Highest Rated Movies',
        color='rating',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_movies_by_year(movies_df):
    """Plot number of movies by year"""
    setup_style()
    
    year_counts = movies_df['year'].value_counts().sort_index()
    
    fig = px.line(
        x=year_counts.index,
        y=year_counts.values,
        labels={'x': 'Year', 'y': 'Number of Movies'},
        title='Movies Released by Year',
        markers=True
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_director_comparison(movies_df, directors):
    """Compare average ratings across directors"""
    setup_style()
    
    director_data = movies_df[movies_df['director'].isin(directors)]
    
    fig = px.box(
        director_data,
        x='director',
        y='rating',
        labels={'director': 'Director', 'rating': 'Movie Rating'},
        title='Rating Distribution by Director',
        color='director'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig


def plot_genre_ratings_heatmap(movies_df, ratings_df):
    """Create heatmap of average ratings by genre"""
    setup_style()
    
    # Merge and calculate average rating by genre
    merged = movies_df.merge(ratings_df, on='movie_id')
    genre_ratings = merged.groupby('genre')['rating_y'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=genre_ratings.values,
        y=genre_ratings.index,
        orientation='h',
        labels={'x': 'Average User Rating', 'y': 'Genre'},
        title='Average User Rating by Genre',
        color=genre_ratings.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=500)
    
    return fig


def plot_user_activity(ratings_df, top_n=20):
    """Plot most active users"""
    setup_style()
    
    user_activity = ratings_df['user_id'].value_counts().head(top_n)
    
    fig = px.bar(
        x=user_activity.index.astype(str),
        y=user_activity.values,
        labels={'x': 'User ID', 'y': 'Number of Ratings'},
        title=f'Top {top_n} Most Active Users',
        color=user_activity.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig


def create_similarity_matrix_plot(similarity_matrix, movie_titles, n=15):
    """Create heatmap of movie similarity matrix"""
    setup_style()
    
    # Take subset for visualization
    subset = similarity_matrix[:n, :n]
    titles_subset = movie_titles[:n]
    
    fig = go.Figure(data=go.Heatmap(
        z=subset,
        x=titles_subset,
        y=titles_subset,
        colorscale='Viridis',
        colorbar=dict(title='Similarity')
    ))
    
    fig.update_layout(
        title='Movie Content Similarity Matrix',
        xaxis_tickangle=-45,
        height=600,
        width=800
    )
    
    return fig
