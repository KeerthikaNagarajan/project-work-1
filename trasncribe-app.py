import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import whisper
import traceback  # Import for printing detailed traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the path to FFmpeg (optional based on your environment)
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust this path if necessary
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"  # Add FFmpeg to PATH

# Create a directory for uploaded videos if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Whisper model globally
model = whisper.load_model("base")

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Transcribe the video
    transcription = transcribe_video(video_path)

    # Optionally, you can delete the video after processing
    os.remove(video_path)

    return jsonify({"transcription": transcription})

def transcribe_video(video_path):
    audio_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")

    # Extract audio from the video
    extract_audio_from_video(video_path, audio_path)

    # Check if the audio file was created
    if not os.path.exists(audio_path):
        return {"error": "Audio extraction failed."}

    print(f"Transcribing audio from: {audio_path}")

    try:
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        return {"error": "Transcription failed."}, 500  # Return a 500 status code

def extract_audio_from_video(video_path, output_audio_path):
    try:
        print(f"Extracting audio from {video_path} to {output_audio_path}...")
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(output_audio_path)
        print("Audio extracted successfully.")
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        traceback.print_exc()  # Print the full traceback for debugging

if __name__ == '__main__':
    app.run(debug=True)
