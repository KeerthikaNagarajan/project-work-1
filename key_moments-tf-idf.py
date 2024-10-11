# Key words based on TF-IDF Vector and providong Key Moments
import cv2
import numpy as np
import os
import whisper
import moviepy.editor as mp
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_audio(video_path, audio_path='audio.wav'):
    """ Extract audio from video and save it to a file. """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')

def transcribe_audio(audio_path):
    """ Transcribe audio to text using Whisper. """
    model = whisper.load_model("base")  # You can choose a different model size as needed
    result = model.transcribe(audio_path)
    return result['text'], result['segments']  # Return text and segments with timings

def find_automatic_keywords(transcription_text, num_keywords=5):
    """ Automatically find keywords based on TF-IDF analysis. """
    # Clean the transcription text by removing punctuation and converting to lower case
    cleaned_text = re.sub(r'[^\w\s]', '', transcription_text.lower())

    # Define custom stop words
    custom_stop_words = ['a', 'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'as', 'that', 
                         'but', 'at', 'by', 'this', 'it', 'an', 'from', 'not', 'or', 'be', 'are', 
                         'so', 'if', 'more', 'there', 'which', 'up', 'when', 'all', 'about', 'has', 
                         'were', 'my', 'he', 'his', 'her', 'they', 'them', 'these', 'you', 'we']

    # Create a TfidfVectorizer with custom stop words
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=num_keywords)
    X = vectorizer.fit_transform([cleaned_text])  # Fit the model on the cleaned text

    # Get feature names and their corresponding TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()

    # Extract keywords based on their TF-IDF scores
    keywords = sorted(zip(feature_names, tfidf_scores[0]), key=lambda x: x[1], reverse=True)[:num_keywords]
    keywords = [word for word, score in keywords if score > 0]  # Keep only words with non-zero scores
    print(f"Automatically found keywords: {keywords}")
    return keywords

def analyze_transcription(segments, keywords):
    """ Identify key moments based on significant words in the transcription. """
    key_moments = []

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        # Check for presence of keywords
        if any(keyword in text.lower() for keyword in keywords):
            key_moments.append((start_time, end_time))
            print(f"Key moment found: {text.strip()} from {start_time:.2f} to {end_time:.2f}")

    return key_moments

def save_key_frames(video_path, key_moments, output_folder):
    """ Save frames at the start of key moments from the video. """
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for start_time, end_time in key_moments:
        frame_number = int(start_time * fps)  # Convert time to frame number
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the frame position
        ret, frame = video.read()
        if ret:
            output_path = os.path.join(output_folder, f"key_moment_{int(start_time)}.jpg")
            cv2.imwrite(output_path, frame)  # Save the frame as an image
            print(f"Saved key frame: {output_path}")

    video.release()

if __name__ == "__main__":
    video_path = "video1.mp4"  # Change this to your video path
    output_folder = "output_frames"  # Change this to your desired output folder

    # Step 1: Extract audio
    audio_path = "audio.wav"
    extract_audio(video_path, audio_path)

    # Step 2: Transcribe audio
    transcription_text, segments = transcribe_audio(audio_path)

    # Step 3: Automatically find keywords from transcription
    keywords = find_automatic_keywords(transcription_text)

    # Step 4: Analyze transcription to find key moments
    key_moments = analyze_transcription(segments, keywords)

    # Step 5: Save key frames based on detected key moments
    if key_moments:
        save_key_frames(video_path, key_moments, output_folder)
        print("Key frames saved successfully.")
    else:
        print("No key moments found.")
