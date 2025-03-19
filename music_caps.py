import pandas as pd

music_data = pd.read_csv('../data/musiccaps-public.csv')

def recommend_music_by_mood(mood):
    # Filter songs that contain the mood in the aspect_list
    matching_songs = music_data[music_data['aspect_list'].str.contains(mood, case=False, na=False)]
    
    # Select the first 5 matching songs
    recommended_songs = matching_songs.head(5)

    # Print the recommended song links
    if not recommended_songs.empty:
        print("Recommended songs for mood '{}':".format(mood))
        for _, song in recommended_songs.iterrows():
            yt_link = f"https://youtu.be/{song['ytid']}?start={song['start_s']}"
            print(yt_link)
    else:
        print("No songs found for mood '{}'.".format(mood))

# Main function to run the recommender
if __name__ == '__main__':
    user_mood = input("Enter a mood: ")
    recommend_music_by_mood(user_mood)