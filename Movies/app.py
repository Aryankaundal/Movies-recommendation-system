import streamlit as st
import pandas as pd
import numpy as np
from lib.movie_recommender import MovieRecommender
from lib.visualizations import (
    plot_genre_distribution,
    plot_ratings_distribution,
    plot_top_rated_movies,
    plot_movies_by_year,
    plot_director_comparison,
    plot_genre_ratings_heatmap,
    plot_user_activity,
    create_similarity_matrix_plot
)
from lib.tmdb_api import fetch_movies_from_api
import os

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Soft dark theme with subtle gradients instead of bright animated background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 0rem 1rem;
    }
    
    /* Content container with dark glass effect */
    .block-container {
        background: rgba(26, 32, 46, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar with soft dark gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background: transparent;
        border-radius: 8px;
        color: #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 172, 254, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(79, 172, 254, 0.3);
    }
    
    /* Text colors for dark theme */
    h1, h2, h3, h4, h5, h6 {
        color: #4facfe !important;
    }
    
    p, span, label, div {
        color: #e0e0e0 !important;
    }
    
    /* Movie card with soft dark styling */
    .movie-card {
        background: rgba(42, 52, 84, 0.6);
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(79, 172, 254, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        border-color: rgba(79, 172, 254, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #1a1a2e;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(79, 172, 254, 0.5);
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        color: #4facfe !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(79, 172, 254, 0.2);
        color: #e0e0e0 !important;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput, .stSlider {
        color: #e0e0e0;
    }
    
    input, select {
        background-color: rgba(26, 32, 46, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
    }
    
    /* Container styling */
    [data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    # Check if data files exist locally
    data_dir_movies = os.path.exists('data/movies.csv')
    data_dir_ratings = os.path.exists('data/ratings.csv')
    root_movies = os.path.exists('movies.csv')
    root_ratings = os.path.exists('ratings.csv')
    
    # Try loading from files first
    if data_dir_movies and data_dir_ratings:
        movies_df = pd.read_csv('data/movies.csv')
        ratings_df = pd.read_csv('data/ratings.csv')
        return movies_df, ratings_df
    elif root_movies and root_ratings:
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
        return movies_df, ratings_df
    else:
        # Fetch from API if no local files
        return fetch_movies_from_api(pages=25)


@st.cache_resource
def get_recommender(_movies_df, _ratings_df):
    recommender = MovieRecommender(_movies_df, _ratings_df)
    recommender.build_content_based_model()
    recommender.build_collaborative_model()
    return recommender


def display_movie_card(movie_row, show_score=False, score_label="Score"):
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # Display movie poster if available
        if 'poster_path' in movie_row and movie_row['poster_path']:
            poster_url = f"https://image.tmdb.org/t/p/w500{movie_row['poster_path']}"
            st.image(poster_url, width=300)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)
    
    with col2:
        st.markdown(f"### {movie_row['title']}")
        st.write(f"**Year:** {movie_row['year']} | **Genre:** {movie_row['genre']}")
        st.write(f"**Director:** {movie_row['director']}")
        st.write(f"**Duration:** {movie_row['duration']} min | **Rating:** ⭐ {movie_row['rating']:.1f}")
        if 'overview' in movie_row and movie_row['overview']:
            with st.expander("📖 Plot Summary"):
                st.write(movie_row['overview'])
    
    with col3:
        if show_score and score_label in movie_row:
            st.metric(score_label, f"{movie_row[score_label]:.2f}")


def main():
    st.title("Movie Recommendation System")
    
    # Load data
    movies_df, ratings_df = load_data()
    
    if movies_df is None or ratings_df is None or len(movies_df) == 0:
        st.error("Failed to load movie data")
        st.stop()
    
    # Initialize recommender
    recommender = get_recommender(movies_df, ratings_df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Get Recommendations", "Data Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Total Movies", len(movies_df))
    st.sidebar.metric("Total Ratings", len(ratings_df))
    st.sidebar.metric("Total Users", ratings_df['user_id'].nunique())
    
    # Main content based on page selection
    if page == "Home":
        show_home_page(movies_df, ratings_df, recommender)
    
    elif page == "Get Recommendations":
        show_recommendations_page(movies_df, ratings_df, recommender)
    
    elif page == "Data Analysis":
        show_analysis_page(movies_df, ratings_df, recommender)


def show_home_page(movies_df, ratings_df, recommender):
    st.header("Welcome to the Movie Recommendation System!")
    
    st.markdown("""
   This system delivers intelligent movie suggestions using modern recommendation strategies:

Preference Profiling: Understands your taste patterns from past selections

Similarity Matching: Finds films that align closely with your favorite themes and styles

Adaptive Ranking: Continuously improves recommendations as you interact

Built using authentic movie metadata sourced from TMDb.
    """)
    
    st.markdown("---")
    
    # Trending movies
    st.subheader("Trending Movies")
    
    # Add sorting controls
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Trending Score", "Year (Newest)", "Year (Oldest)", "Rating"],
            key="trending_sort"
        )
    
    try:
        trending_movies = recommender.get_trending_movies(n=10)
        display_column = 'trending_score' if 'trending_score' in trending_movies.columns else 'popularity'
    except:
        trending_movies = recommender.get_popular_movies(n=10)
        display_column = 'num_ratings'
    
    # Apply sorting based on selection
    if sort_by == "Year (Newest)":
        trending_movies = trending_movies.sort_values('year', ascending=False)
    elif sort_by == "Year (Oldest)":
        trending_movies = trending_movies.sort_values('year', ascending=True)
    elif sort_by == "Rating":
        trending_movies = trending_movies.sort_values('rating', ascending=False)
    # Default is already sorted by trending score/popularity
    
    for idx, row in trending_movies.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
            with col1:
                # Display small poster thumbnail
                if 'poster_path' in row and row['poster_path']:
                    poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"
                    st.image(poster_url, width=300)
            with col2:
                st.write(f"**{row['title']}** ({int(row['year'])})")
            with col3:
                st.write(f"⭐ {row['rating']:.1f}")
            with col4:
                st.write(f"🎭 {row['genre']}")
            with col5:
                if display_column in row:
                    if display_column == 'num_ratings':
                        st.write(f"📊 {int(row[display_column])} ratings")
                    elif display_column == 'popularity':
                        st.write(f"🔥 {int(row[display_column])}")
                    else:
                        st.write(f"🔥 {int(row[display_column])}")
    
    st.markdown("---")
    
    # Popular movies
    st.subheader("Most Popular Movies")
    
    # Add sorting controls for popular movies
    col1, col2 = st.columns([2, 1])
    with col1:
        popular_sort_by = st.selectbox(
            "Sort by:",
            ["Rating", "Popularity", "Year (Newest)", "Year (Oldest)"],
            key="popular_sort"
        )
    
    popular_movies = recommender.get_popular_movies(n=10)
    
    # Apply sorting based on selection
    if popular_sort_by == "Year (Newest)":
        popular_movies = popular_movies.sort_values('year', ascending=False)
    elif popular_sort_by == "Year (Oldest)":
        popular_movies = popular_movies.sort_values('year', ascending=True)
    elif popular_sort_by == "Rating":
        popular_movies = popular_movies.sort_values('rating', ascending=False)
    elif popular_sort_by == "Popularity":
        popular_movies = popular_movies.sort_values('num_ratings', ascending=False)
    
    for idx, row in popular_movies.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
            with col1:
                # Display small poster thumbnail
                if 'poster_path' in row and row['poster_path']:
                    poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"
                    st.image(poster_url, width=300)
            with col2:
                st.write(f"**{row['title']}** ({row['year']})")
            with col3:
                st.write(f"⭐ {row['rating']:.1f}")
            with col4:
                st.write(f"🎭 {row['genre']}")
            with col5:
                st.write(f"📊 {int(row['num_ratings'])} ratings")
    
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Rating", f"{ratings_df['rating'].mean():.2f}")
    
    with col2:
        most_common_genre = movies_df['genre'].mode()[0]
        st.metric("Most Common Genre", most_common_genre)
    
    with col3:
        avg_duration = movies_df['duration'].mean()
        st.metric("Avg Movie Length", f"{int(avg_duration)} min")
    
    with col4:
        year_range = f"{movies_df['year'].min()}-{movies_df['year'].max()}"
        st.metric("Year Range", year_range)


def show_recommendations_page(movies_df, ratings_df, recommender):
    """Recommendations page"""
    st.header("Get Personalized Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "By Genre"])
    
    with tab1:
        st.subheader("Content-Based Recommendations")
        st.write("Get movies similar to one you like based on genre, director, and features.")
        
        selected_movie = st.selectbox(
            "Select a movie you like:",
            movies_df['title'].tolist(),
            key="content_movie"
        )
        
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10, key="content_slider")
        
        if st.button("Get Similar Movies", key="content_btn"):
            movie_id = movies_df[movies_df['title'] == selected_movie]['movie_id'].values[0]
            recommendations = recommender.get_content_recommendations(movie_id, n=num_recommendations)
            
            st.success(f"Movies similar to '{selected_movie}':")
            
            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['title']} (Similarity: {row['similarity_score']:.2f})"):
                    display_movie_card(row, show_score=True, score_label='similarity_score')
    
    with tab2:
        st.subheader("Collaborative Filtering Recommendations")
        st.write("Get recommendations based on users with similar taste.")
        
        user_id = st.number_input(
            "Enter User ID (1-100):",
            min_value=1,
            max_value=100,
            value=1,
            key="collab_user"
        )
        
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10, key="collab_slider")
        
        if st.button("Get Recommendations", key="collab_btn"):
            # Show user's rated movies
            user_ratings = ratings_df[ratings_df['user_id'] == user_id].merge(
                movies_df, on='movie_id'
            ).sort_values('rating_x', ascending=False)
            
            st.write(f"**User {user_id}'s top rated movies:**")
            top_user_movies = user_ratings.head(5)
            for _, row in top_user_movies.iterrows():
                st.write(f"- {row['title']} ({row['rating_x']}/5)")
            
            st.markdown("---")
            
            recommendations = recommender.get_collaborative_recommendations(user_id, n=num_recommendations)
            
            if len(recommendations) > 0:
                st.success("Recommended for you:")
                
                for idx, row in recommendations.iterrows():
                    with st.expander(f"{row['title']}"):
                        display_movie_card(row)
            else:
                st.warning("Not enough data to generate recommendations for this user.")
    
    with tab3:
        st.subheader("Recommendations by Genre")
        st.write("Browse top-rated movies in your favorite genre.")
        
        genres = sorted(movies_df['genre'].unique())
        selected_genre = st.selectbox("Select a genre:", genres, key="genre_select")
        
        if st.button("Show Movies", key="genre_btn"):
            genre_movies = recommender.get_movies_by_genre(selected_genre).head(15)
            
            st.success(f"Top {selected_genre} movies:")
            
            for idx, row in genre_movies.iterrows():
                with st.expander(f"{row['title']} ({row['rating']:.1f})"):
                    display_movie_card(row)


def show_analysis_page(movies_df, ratings_df, recommender):
    """Data analysis and visualization page"""
    st.header("Data Analysis & Insights")
    
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Rating Insights", "Advanced Analytics"])
    
    with tab1:
        st.subheader("Movie Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = plot_genre_distribution(movies_df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = plot_movies_by_year(movies_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        fig3 = plot_top_rated_movies(movies_df, n=15)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Rating Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = plot_ratings_distribution(ratings_df)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = plot_user_activity(ratings_df, top_n=20)
            st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("---")
        
        fig6 = plot_genre_ratings_heatmap(movies_df, ratings_df)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.subheader("Advanced Analytics")
        
        # Director comparison
        st.write("**Compare Directors**")
        top_directors = movies_df['director'].value_counts().head(10).index.tolist()
        selected_directors = st.multiselect(
            "Select directors to compare:",
            top_directors,
            default=top_directors[:5]
        )
        
        if selected_directors:
            fig7 = plot_director_comparison(movies_df, selected_directors)
            st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("---")
        
        # Similarity matrix
        st.write("**Content Similarity Matrix**")
        if st.button("Generate Similarity Matrix"):
            if recommender.content_similarity is not None:
                fig8 = create_similarity_matrix_plot(
                    recommender.content_similarity,
                    movies_df['title'].tolist(),
                    n=15
                )
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.error("Similarity matrix not available")


if __name__ == "__main__":
    main()
