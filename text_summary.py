from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

# Example usage
video_path = "video1.mp4"  # Path to your video file
audio_path = "extracted_audio.wav"  # Path to save extracted audio
extract_audio_from_video(video_path, audio_path)
print("Audio extracted successfully.")

import whisper

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Example usage
transcription = transcribe_audio(audio_path)
print("Transcription:", transcription)
with open("extracted_text.txt", "w") as text_file:
    text_file.write(transcription)

print("Transcription saved to extracted_text.txt")

pip install transformers

from transformers import pipeline

# Specify the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_transcription(transcription):
    summary = summarizer(transcription, max_length=50, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# Example usage
summary = summarize_transcription(transcription)
print("Summary:", summary)


from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the DistilBERT model and tokenizer specifically fine-tuned for question answering
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Load your text data (e.g., from a text file)
with open('extracted_text.txt', 'r') as file:
    context = file.read()

def answer_question(question):
    # Encode the question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    
    # Get the input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Forward pass through the model to get start and end scores
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Get the most likely start and end token positions
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1  # +1 because the end index is inclusive
    
    # Decode the answer from the context
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Main loop for question answering
while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = answer_question(question)
    print("Answer:", answer)

