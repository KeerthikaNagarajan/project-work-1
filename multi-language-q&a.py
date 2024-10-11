#Answer Text to Speech in multiple languages.
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, MarianMTModel, MarianTokenizer
import torch
from gtts import gTTS
import os
from playsound import playsound

# Function to handle text-to-speech in different languages
def text_to_speech(text, language_code='en'):
    tts = gTTS(text, lang=language_code)
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

# Function to answer questions
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

# Function to load MarianMT model and tokenizer for translation
def load_translation_model(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Function to translate text using MarianMT
def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Main loop for question answering with translation and text-to-speech
while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    
    # Get the answer
    answer = answer_question(question)
    print("Answer:", answer)
    
    # Ask if the user wants to translate the answer
    translate = input("Do you want to translate the answer? (yes/no): ").lower()
    
    if translate == 'yes':
        # Ask for target language
        target_lang = input("Enter target language code (e.g., 'fr' for French, 'es' for Spanish): ").lower()
        source_lang = 'en'  # Assuming the context is in English
        
        # Load the MarianMT model for translation
        translation_model, translation_tokenizer = load_translation_model(source_lang, target_lang)
        
        # Translate the answer
        translated_answer = translate_text(answer, translation_model, translation_tokenizer)
        print(f"Translated Answer ({target_lang}):", translated_answer)
        
        # Convert the translated answer to speech
        text_to_speech(translated_answer, target_lang)
    else:
        # Convert the original answer to speech
        text_to_speech(answer)
