i dont know how to push jup note in git sorry mate
# project-work-1
```py
import openai
from moviepy.editor import VideoFileClip
import whisper
import time

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual API key

def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)
    return output_audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def summarize_text(text):
    retries = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Change to gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Summarize this text: {text}"}
                ],
                max_tokens=150  # Adjust token limit as needed
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            retries += 1
            wait_time = 10 * retries  # Increase wait time on each retry
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying

# Example usage
video_path = "video.mp4"  # Path to your video file
audio_path = "extracted_audio.wav"  # Path to save extracted audio

# Extract, Transcribe, and Summarize
extract_audio_from_video(video_path, audio_path)
transcription = transcribe_audio(audio_path)
summary = summarize_text(transcription)
print(summary)
```
