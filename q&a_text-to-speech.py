from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from gtts import gTTS
import os
from playsound import playsound

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_file = "answer.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)  # Remove the audio file after playing it

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
    
    # Convert the answer to speech
    text_to_speech(answer)
