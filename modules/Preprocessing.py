import json
import re
import csv
import re
import nltk
import emoji
import csv
import pandas as pd
import numpy as np
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Load the JSON file into a Python dictionary
with open('data/emoticon_dict.json', 'r') as file:
    emotion_dict = json.load(file)

def replace_emoticons_with_emotions(text, emotion_dict):
    for emoticon, emotion in emotion_dict.items():
        text = text.replace(emoticon, emotion)
    return text

def convert_emojis_to_text(text):
    return emoji.demojize(text)

def contains_emoji(text):
    # Create a regular expression for emojis
    emoji_regex = emoji.get_emoji_regexp()

    # Check if the text contains an emoji
    if emoji_regex.search(text):
        return True
    else:
        return False

def create_dict_from_csv(file_name):
    dictionary = {}
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # Skip the header row
        for row in csv_reader:
            key = row[0]
            value = row[1]
            dictionary[key] = value
    return dictionary

def expand_abbreviations(text):
    abbreviations_dict = create_dict_from_csv('data/abbv.csv')
    for abbr, full_form in abbreviations_dict.items():
        # Use regex to match the abbreviation as a whole word to avoid partial matches
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)
    return text


def custom_correct_spelling(text, correct_words):
    for word in correct_words:
        if word in text:
            text = text.replace(word, correct_words[word])
    return text

correct_words = {"hapy": "happy"}

import re

def remove_repeated_chars(s):
    # Match a character followed by at least 2 repetitions and replace with just two repetitions
    return re.sub(r'(\w)\1{2,}', r'\1\1', s)

def remove_spaces(text):
    # Regex pattern to find spaces between letters
    pattern = r"(?<=\b\w) (?=\w\b)"
    # Using re.sub to replace the matched spaces with an empty string
    text = re.sub(pattern, '', text)
    return text

def remove_hyphenated_capitalized_words(text):
    pattern = r'\b([A-Z][a-z]*-)*([A-Z][a-z]*)\b'
    result = re.sub(pattern, lambda match: match.group(0).replace('-', ''), text)
    return result

def normalize_hashtags(hashtags):
    processed_hashtags = []
    for hashtag in hashtags:
        # Insert spaces before each capital letter
        processed_hashtag = re.sub(r'([A-Z])', r' \1', hashtag)
        processed_hashtags.append(processed_hashtag)
    return processed_hashtags

def preprocess_text(text):
    
    # Remove hashtags and seperate each word
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text)
    processed_hashtags = normalize_hashtags(hashtags)
    for i, hashtag in enumerate(hashtags):
        text = text.replace(hashtag, processed_hashtags[i])
        
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+|\S+\.ly|\S+\.ph|\S+\.net', '', text, flags=re.MULTILINE)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove newline characters
    text = text.replace(r"\n", " ")

    # Replace &amp; to and characters
    text = text.replace(r"&amp;", "and")
    
    text = re.sub(r'\.', ' ', text)    
    
    #remove hyphenated
    text = remove_hyphenated_capitalized_words(text)

    #Remove all caps
    text = text.lower()

    # Expand Abbrevations
    text = expand_abbreviations(text)

    # Remove text emoticons like :)))) to their emotion equivalence
    text = replace_emoticons_with_emotions(text,emotion_dict)

    # Remove Spaces between letters like e v e r y t h i n g
    text = remove_spaces(text)
    
    # Fix contractions from don't to do not
    text = contractions.fix(text)

    # Remove special characters and numbers
    text = re.sub(r'\/', ' or ', text)
    text = re.sub(r'\W', ' ', text)

    # # Remove repeated characters like happppyyyyy to hapy
    text= remove_repeated_chars(text)

    # # Corrects the word to its correct spelling to happy
    text = custom_correct_spelling(text, correct_words)

    # Replace nan to "" characters
    text = text.replace(r"nan", "")
    
    # Convert emojis to textual representation
    if contains_emoji:
        text = convert_emojis_to_text(text)
    
    return text


def load_stopwords_from_csv(file_path):
    stopwords_list = pd.read_csv(file_path)['word'].tolist()
    return stopwords_list

def customize_stopwords(stopwords):
    # Load the custom stopwords from CSV files
    interjections = load_stopwords_from_csv('algo/interjections.csv')
    negations = load_stopwords_from_csv('algo/negations.csv')
    amplifiers = load_stopwords_from_csv('algo/amplifiers.csv')

    # Combine the custom stopwords into a single list
    custom_stopwords = interjections + negations + amplifiers
    # Remove the custom stopwords from the NLTK stopwords list
    filtered_stopwords = [word for word in stopwords if word not in custom_stopwords]

    return filtered_stopwords

# Initialize the DistilBert tokenizer
lemmatizer = WordNetLemmatizer()
emotionlex_df = pd.read_csv('algo/emolex_words.csv')

def tokenize(text):
    # Tokenize the text into words
    word_tokens = word_tokenize(text)

    nltk_stopwords = stopwords.words('english')
    customized_stopwords = customize_stopwords(nltk_stopwords)

    filtered_tokens = [token for token in word_tokens if token not in customized_stopwords]

    lemmatized_tokens = []
    for word in filtered_tokens:
        if word in emotionlex_df['word'].tolist():  # Check if word exists in NRC emotion lexicon (assuming 'word' is the column name)
            lemmatized_tokens.append(word)  # Don't lemmatize, keep the original word
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))  # Lemmatize other words

    return lemmatized_tokens


