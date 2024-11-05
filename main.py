from recommender import MusicRecommender
import traceback
import requests

def print_mood_info():
    """Print available moods and their descriptions"""
    print("\nAvailable Moods:")
    print("- happy (upbeat, cheerful, energetic)")
    print("- sad (melancholic, emotional, gloomy)")
    print("- relaxing (calm, peaceful, mellow)")
    print("- romantic (love songs, sweet, beautiful)")
    print("- angry (aggressive, intense, heavy)")
    print("- energetic (powerful, dynamic, lively)")

def main():
    try:
        print("Initializing Music Recommendation System...")
        recommender = MusicRecommender()
        
        print("Loading LastFM dataset (this might take a few minutes)...")
        recommender.load_data()
        
        while True:
            print("\nMusic Recommendation System")
            print("1. Find similar songs by artist")
            print("2. Recommend songs by mood")
            print("3. Recommend songs by tag/genre")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '4':
                print("Thank you for using the Music Recommender!")
                break
            
            elif choice == '1':
                artist = input("Enter artist name: ")
                results = recommender.recommend_songs('artist', artist)
                
                if not results.empty:
                    print("\nRecommended Songs:")
                    for _, row in results.iterrows():
                        print(f"\nSong: {row['song_name']}")
                        print(f"Artist: {row['artist_name']}")
                        print(f"Similarity Score: {row['similarity']:.2f}")
                        print(f"URL: {row['url']}")
                        youtube_search_link = f"https://www.youtube.com/results?search_query={requests.utils.quote(row['song_name'] + ' ' + row['artist_name'])}"
                        print(f"YouTube Search Link: {youtube_search_link}")
            
            elif choice == '2':
                print_mood_info()
                mood = input("\nEnter mood: ")
                results = recommender.recommend_songs('mood', mood)
                
                if not results.empty:
                    print(f"\nRecommended {mood.title()} Songs:")
                    for _, row in results.iterrows():
                        print(f"\nSong: {row['song_name']}")
                        print(f"Artist: {row['artist_name']}")
                        print(f"Mood Score: {row['mood_score']:.2f}")
                        print(f"URL: {row['url']}")
                        youtube_search_link = f"https://www.youtube.com/results?search_query={requests.utils.quote(row['song_name'] + ' ' + row['artist_name'])}"
                        print(f"YouTube Search Link: {youtube_search_link}")
                        
            
            elif choice == '3':
                print("\nPopular tags: rock, electronic, pop, jazz, metal")
                tag = input("Enter tag/genre: ")
                results = recommender.recommend_songs('tag', tag)
                
                if not results.empty:
                    print("\nRecommended Songs:")
                    for _, row in results.iterrows():
                        print(f"\nSong: {row['song_name']}")
                        print(f"Artist: {row['artist_name']}")
                        print(f"URL: {row['url']}")
                        
                        youtube_search_link = f"https://www.youtube.com/results?search_query={requests.utils.quote(row['song_name'] + ' ' + row['artist_name'])}"
                        print(f"YouTube Search Link: {youtube_search_link}")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()