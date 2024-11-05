import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import zipfile
import io
import os
from tqdm import tqdm

class MusicRecommender:
    def __init__(self):
        self.songs_df = None
        self.artists_df = None
        self.tags_df = None
        self.user_artists_df = None
        self.artist_tags_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        
        # Define mood categories and their related tags
        self.mood_categories = {
            'happy': ['happy', 'upbeat', 'fun', 'cheerful', 'joyful', 'energetic', 'party'],
            'sad': ['sad', 'melancholic', 'depressing', 'gloomy', 'dark', 'emotional'],
            'relaxing': ['chill', 'relaxing', 'calm', 'peaceful', 'mellow', 'ambient'],
            'romantic': ['romantic', 'love', 'lovely', 'sweet', 'beautiful'],
            'angry': ['angry', 'aggressive', 'intense', 'heavy', 'rage'],
            'energetic': ['energetic', 'upbeat', 'powerful', 'dynamic', 'lively']
        }

    def download_dataset(self):
        """Download LastFM dataset"""
        print("Downloading LastFM dataset...")
        url = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
        
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                response = requests.get(url)
                with open("data/lastfm.zip", 'wb') as f:
                    f.write(response.content)
            
            print("Extracting files...")
            with zipfile.ZipFile("data/lastfm.zip", 'r') as zip_ref:
                zip_ref.extractall("data")
            
            os.remove("data/lastfm.zip")
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

    def load_data(self):
        """Load LastFM dataset and create song-level data"""
        if not os.path.exists("data/artists.dat"):
            print("Dataset not found, downloading...")
            self.download_dataset()
        
        print("Loading dataset...")
        try:
            # Load base datasets
            self.artists_df = pd.read_csv("data/artists.dat", 
                                        sep='\t', 
                                        encoding='latin-1')
            
            self.user_artists_df = pd.read_csv("data/user_artists.dat",
                                             sep='\t',
                                             encoding='latin-1')
            
            self.tags_df = pd.read_csv("data/tags.dat",
                                     sep='\t',
                                     encoding='latin-1')
            
            self.artist_tags_df = pd.read_csv("data/user_taggedartists.dat",
                                            sep='\t',
                                            encoding='latin-1')
            
            print("Creating song-level dataset...")
            self.create_song_dataset()
            
            print("\nDataset loaded successfully!")
            print(f"Total Songs: {len(self.songs_df)}")
            print(f"Total Artists: {len(self.artists_df)}")
            print("Columns in songs_df:", self.songs_df.columns.tolist())
            
            self.process_song_features()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    # def create_youtube_search_link(self, song_name, artist_name):
    #     search_query = f"{song_name} {artist_name}"
    #     return f"https://www.youtube.com/results?search_query={requests.utils.quote(search_query)}"

    def create_song_dataset(self):
        """Create a song-level dataset with popular songs for each artist"""
        # Create a base song dataset with multiple songs per artist
        songs_data = []
        
        # Get top tags for each artist
        artist_tags = pd.merge(self.artist_tags_df, self.tags_df, on='tagID')
        artist_tags_grouped = artist_tags.groupby('artistID')['tagValue'].agg(list).reset_index()
        
        # Create multiple songs for each artist based on their tags and popularity
        for _, artist in self.artists_df.iterrows():
            artist_id = artist['id']
            
            # Get artist's tags
            tags = artist_tags_grouped[
                artist_tags_grouped['artistID'] == artist_id
            ]['tagValue'].iloc[0] if len(
                artist_tags_grouped[artist_tags_grouped['artistID'] == artist_id]
            ) > 0 else []
            
            # Generate song names based on common song types
            song_types = [
                'Greatest Hits', 'Live Performance', 'Acoustic Version',
                'Radio Edit', 'Album Version', 'Single Version',
                'Remix', 'Extended Mix', 'Studio Recording'
            ]
            
            # Create multiple songs for the artist
            for song_type in song_types:
                song_name = f"{artist['name']} - {song_type}"
                
                # Generate song features
                tempo = np.random.uniform(60, 180)  # BPM
                energy = np.random.uniform(0, 1)
                danceability = np.random.uniform(0, 1)
                valence = np.random.uniform(0, 1)  # Musical positiveness
                
                # youtube_search_link = self.create_youtube_search_link(song_name, artist['name'])
                
                songs_data.append({
                    'song_name': song_name,
                    'artist_id': artist_id,
                    'artist_name': artist['name'],
                    'tags': tags,
                    'tempo': tempo,
                    'energy': energy,
                    'danceability': danceability,
                    'valence': valence,
                    'url': artist['url'],
                    'youtube_search_link': youtube_search_link
                })
        
        self.songs_df = pd.DataFrame(songs_data)

    def process_song_features(self):
        """Process song features and create feature matrix"""
        print("Processing song features...")
        
        # Create text features from tags
        self.songs_df['tag_text'] = self.songs_df['tags'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Create TF-IDF matrix from tags
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.songs_df['tag_text'])
        
        # Process mood scores for songs
        for mood in self.mood_categories.keys():
            self.songs_df[f'mood_{mood}'] = self.songs_df.apply(
                lambda x: self._calculate_song_mood_score(x, mood),
                axis=1
            )

    def _calculate_song_mood_score(self, song, mood):
        """Calculate mood score for a song based on its features and tags"""
        # Initialize base score from audio features
        score = 0
        
        # Add score based on audio features
        if mood in ['happy', 'energetic']:
            score += (song['energy'] * 0.3 + song['valence'] * 0.4 + 
                     song['danceability'] * 0.3)
        elif mood == 'sad':
            score += ((1 - song['valence']) * 0.5 + (1 - song['energy']) * 0.3 + 
                     (1 - song['danceability']) * 0.2)
        elif mood == 'relaxing':
            score += ((1 - song['energy']) * 0.4 + song['valence'] * 0.3 + 
                     (1 - song['tempo']/180) * 0.3)
        else:
            score += (song['energy'] * 0.4 + song['valence'] * 0.3 + 
                     song['danceability'] * 0.3)
        
        # Add score based on tags
        if isinstance(song['tags'], list):
            mood_keywords = self.mood_categories[mood]
            tag_score = sum(
                1 for tag in song['tags'] 
                if any(keyword in tag.lower() for keyword in mood_keywords)
            ) / max(len(song['tags']), 1)
            score = 0.7 * score + 0.3 * tag_score
        
        return score

    def recommend_songs(self, query_type, query_value, n_recommendations=5):
        """Recommend songs based on query type and value"""
        try:
            if query_type == 'mood':
                return self.recommend_by_mood(query_value, n_recommendations)
            elif query_type == 'artist':
                return self.recommend_similar_songs(query_value, n_recommendations)
            elif query_type == 'tag':
                return self.recommend_by_tag(query_value, n_recommendations)
            else:
                print(f"Unknown query type: {query_type}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error recommending songs: {str(e)}")
            return pd.DataFrame()

    def recommend_by_mood(self, mood, n_recommendations=5):
        """Recommend songs based on mood"""
        if mood.lower() not in self.mood_categories:
            print(f"Unknown mood. Available moods: {list(self.mood_categories.keys())}")
            return pd.DataFrame()
        
        mood_col = f'mood_{mood.lower()}'
        
        # Get songs sorted by mood score and add variety
        recommendations = self.songs_df.copy()
        recommendations['mood_score'] = recommendations[mood_col]
        
        # Add some randomness to promote variety
        recommendations['random_boost'] = np.random.uniform(0, 0.2, len(recommendations))
        recommendations['final_score'] = recommendations['mood_score'] + recommendations['random_boost']
        
        return recommendations.nlargest(n_recommendations, 'final_score')[
            ['song_name', 'artist_name', 'mood_score', 'url']
        ]

    def recommend_similar_songs(self, artist_name, n_recommendations=5):
        """Recommend similar songs based on artist and song features"""
        artist_songs = self.songs_df[
            self.songs_df['artist_name'].str.contains(artist_name, case=False)
        ]
        
        if len(artist_songs) == 0:
            print(f"No songs found for artist: {artist_name}")
            return pd.DataFrame()
        
        # Get a random song from the artist as reference
        reference_song = artist_songs.sample(1).index[0]
        
        # Calculate similarities
        similarities = cosine_similarity(
            self.tfidf_matrix[reference_song:reference_song+1],
            self.tfidf_matrix
        )[0]
        
        # Get recommendations excluding the same artist
        recommendations = self.songs_df.copy()
        recommendations['similarity'] = similarities
        recommendations = recommendations[
            recommendations['artist_name'].str.lower() != artist_name.lower()
        ]
        
        return recommendations.nlargest(n_recommendations, 'similarity')[
            ['song_name', 'artist_name', 'similarity', 'url', 'youtube_search_link']
        ]

    def recommend_by_tag(self, tag, n_recommendations=5):
        """Recommend songs based on tag/genre"""
        # Find songs with matching tags
        recommendations = self.songs_df[
            self.songs_df['tag_text'].str.contains(tag, case=False)
        ].copy()
        
        if len(recommendations) == 0:
            print(f"No songs found with tag: {tag}")
            return pd.DataFrame()
        
        # Add some randomness for variety
        recommendations['random_score'] = np.random.uniform(0, 1, len(recommendations))
        
        return recommendations.nlargest(n_recommendations, 'random_score')[
            ['song_name', 'artist_name', 'url']
        ]