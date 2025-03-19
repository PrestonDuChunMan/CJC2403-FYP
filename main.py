import tkinter as tk
import pandas as pd
from tkinter import filedialog, messagebox
from gemini_analysis import gemini_analysis
from video_intelligence_analysis import vi_analysis
from dotenv import load_dotenv
from recommender import MusicRecommender
import threading 
# from MusicGen.musicGen import generate_music_tensors #part3.2 
from music_caps import recommend_music_by_mood

load_dotenv()

def process_video(video_path, music_choice):
    # Step 1: Analyze the mood of the video using Gemini
    mood = gemini_analysis(video_path)
    
    if mood is None:
        print("Failed to analyze mood.")
        return

    print(f"Gemini Mood Analysis Result: {mood}")
    
    # Step 2: Handle music choice based on user selection
    if music_choice == "vocal":
        # Step 3.1: Only initialize the recommender if user chooses vocal music
        print("Initializing recommender for vocal music...")
        
        recommendation_thread = threading.Thread(target=process_vocal_music, args=(mood,))
        recommendation_thread.start()
        
        # Wait for the thread to finish before the main program ends
        recommendation_thread.join()
        

    elif music_choice == "instrumental":
        # Step 3.2: Use instrumental music dataset from musicCaps
        print("Using instrumental music dataset from musicCaps.")
        
        # Call musicCaps recommend function to get instrumental music based on the video mood
        recommend_music_by_mood(mood)
    
    elif music_choice == "ai-generated":
        # Step 3.3: Use VI for shot change detection and object detection
        print("Generating AI-generated music based on video analysis.")
        # Your code for AI-generated music and video analysis here (e.g., shot change detection and object detection)
    
    else:
        print("Invalid music choice. Please select from 'vocal', 'instrumental', or 'ai-generated'.")

def process_vocal_music(mood):
    try:
        # Initialize the recommender object
        recommender = MusicRecommender()
        recommender.load_data()  # Load the LastFM dataset
        
        # Get vocal music recommendations for the given mood
        print(f"Fetching vocal music recommendations for mood: {mood}")
        recommendations = recommender.recommend_by_mood(mood)
        
        if recommendations.empty:
            print(f"No vocal music recommendations found for mood '{mood}'.")
        else:
            print("Recommended vocal music:")
            for _, rec in recommendations.iterrows():
                print(f"{rec['song_name']} by {rec['artist_name']} - {rec['url']}")
    except Exception as e:
        print(f"Error during vocal music recommendation: {e}")


if __name__ == "__main__":
    video_path = input("Please upload your MP4 video file: ")
    print("Choose music type: vocal, instrumental, or ai-generated.")
    music_choice = input("Enter music choice: ").strip().lower()
    
    # Step 1: Process the video based on the user's choice
    process_video(video_path, music_choice)
