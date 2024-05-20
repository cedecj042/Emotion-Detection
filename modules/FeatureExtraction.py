import pandas as pd
import nltk , re
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download('punkt')

interjections_df = pd.read_csv('algo/interjections.csv')
negations_df = pd.read_csv('algo/negations.csv')
amplifiers_df = pd.read_csv('algo/amplifiers.csv')
emotionlex_df = pd.read_csv('algo/emolex_words.csv')
emointensity_df = pd.read_csv('algo/emolex_intensity.csv')

def get_emotion(word):
    emolex = emotionlex_df[emotionlex_df["word"] == word]
    if not emolex.empty:
        return True, emolex.iloc[:,1:9].values
    else:
        return False, 0

def detect_negation(word):
    # Check if the word is in the negation or amplifier dataframes
    negation = negations_df[negations_df['word'] == word]
    # If the word is found in either dataframe, it's a negation or an amplifier
    if not negation.empty:
        return True, negation['score'].values  # Return True and the negation score
    else:
        return False, 0 


def detect_amplifier(word):
    amplifier = amplifiers_df[amplifiers_df['word'] == word]
    # If the word is found in either dataframe, it's a negation or an amplifier
    if not amplifier.empty:
        return True, amplifier['score'].values # Return True and the amplifier score
    else:
        return False, 0 
def detect_interjection(word):
    # Convert the 'word' to lowercase before searching
    word = word.lower()
    # Convert the 'word' column in the DataFrame to lowercase for comparison
    interjections_df['word'] = interjections_df['word'].str.lower()
    interjection = interjections_df[interjections_df['word'] == word]
    if not interjection.empty:
        return True, interjection[['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']].values
    return False, 0

def mapout_negatives(arr):
    # Define the mapping of each index to its opposite index
    index_mapping = {0: 3, 1: 6, 2: 7, 3: 0, 4: 5, 5: 4, 6: 1, 7: 2}

    # Iterate through the array
    for i in range(len(arr)):
        # Check if the current value is negative
        if arr[i] < 0:
            # Add the absolute value of the current value to its opposite index
            arr[index_mapping[i]] += abs(arr[i])
            # Set the current value to 0
            arr[i] = 0.0
    return arr

def normalize_array(arr):
    # Min-max scaling
    new_arr = mapout_negatives(arr)
    min_value = min(new_arr)
    max_value = max(new_arr)

    if not min_value and not max_value:
        return arr

    return [(value - min_value) / (max_value - min_value) for value in new_arr]

def get_average(arr,len):
    for i in range(len):
        arr[i] = arr[i] / len
    return arr

def get_emotion_intensity(word):
    intensity = emointensity_df[emointensity_df["word"] == word]
    if not intensity.empty:
        # Ensure the returned array has a fixed length, e.g., 8 for 8 emotions
        return np.array(intensity.iloc[:,1:9].values, dtype=float)
    else:
        # Return an array of zeros with a fixed length
        return np.zeros(8, dtype=float)
        

def get_emotion_score(text):
    sentence = word_tokenize(text)
    total_score = np.zeros(8)  # Initialize total score array
    multiplier = []  # Initialize multiplier as a list
    inter_arr = np.zeros(8) 

    for word in sentence:
        is_neg, neg_score = detect_negation(word)
        if not is_neg:
            is_amp, amp_score = detect_amplifier(word)
            if not is_amp:
                is_emotion, scores = get_emotion(word)
                if not is_emotion:
                    multiplier = []
                else:
                    temp_score = []
                    new_score = []
                    intensity_score = get_emotion_intensity(word)
                    new_score = np.add(intensity_score, scores[0])

                    new_score = new_score.flatten()
                    for x in range(len(new_score)):
                        if new_score[x] == 0:
                            temp_score.append(0)
                            continue
                        # Initialize previous_score
                        else:
                            if multiplier:  # Ensure multiplier is not empty
                                previous_score = multiplier[0][list(multiplier[0].keys())[0]]
                                previous_element = multiplier[0]
                                if len(multiplier) > 1:
                                    for i in range(1,len(multiplier)):
                                        if 'neg' in multiplier[i] and 'neg' in previous_element:
                                            previous_score = multiplier[i]['neg'] + previous_score
                                        elif 'amp' in multiplier[i] and 'amp' in previous_element:
                                            previous_score = (multiplier[i]['amp'] + previous_score) / 2
                                        elif 'amp' in multiplier[i] and 'neg' in previous_element:
                                            previous_score = (multiplier[i]['amp'] * previous_score ) + previous_score
                                        elif 'neg' in multiplier[i] and 'amp' in previous_element:
                                            previous_score = (multiplier[i]['neg'] * previous_score) + multiplier[i]['neg']
                                        if i == len(multiplier) - 1:
                                            # print(previous_score , new_score[x], (new_score[x] * previous_score) + new_score[x])
                                            previous_score = (new_score[x] * previous_score) + new_score[x]
                                else:
                                    previous_score = (new_score[x] * previous_score) + new_score[x]
                                temp_score.append(previous_score)
                            else:
                                temp_score.append(0)
                                
                    if np.array(temp_score).sum() == 0:
                        total_score += np.array(new_score)
                    else:
                        total_score += np.array(temp_score)
                    multiplier = []
            else:
                multiplier.append({"amp": amp_score[0]})
        else:
            multiplier.append({"neg": neg_score[0]})

        is_inter, inter_scores = detect_interjection(word)
        if is_inter:
            inter_arr += np.array(inter_scores[0])
    total_score += np.array(get_average(inter_arr,len(inter_arr))) 
    return normalize_array(total_score)
    
def check_emotion_scores(emotion_scores):
    for index, score in enumerate(emotion_scores):
        if not all(0 <= s <= 1 for s in score):
            print(f"Row index: {index}")
            print(f"Emotion score: {score}")
            print("Emotion score is not within the range [0, 1]")
            return
    print("All emotion scores are within the range [0, 1]")

def filter_pos_tags(text):
  desired_tags = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "UH", "NN", "NNS", "NNP", "NNPS"]
  tokens = nltk.word_tokenize(text)
  pos_tags = nltk.pos_tag(tokens)
  filtered_words = [word for word, tag in pos_tags if tag in desired_tags]
  return filtered_words

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to compute unigram frequency distribution
def uni_freq(x, df):
    tmp = ' '.join(df[df[x] == 1]["processed"])
    tmp = re.sub('\n', '', tmp)
    tmp = ' '.join([word for word in word_tokenize(tmp) if word not in stop_words])
    tmp = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(tmp)])

    return FreqDist(nltk.word_tokenize(tmp))

# Function to compute bigram frequency distribution
def bi_freq(x, df):
    tmp = ' '.join(df[df[x] == 1]["processed"])
    tmp = re.sub('\n', '', tmp)
    tmp = ' '.join([word for word in word_tokenize(tmp) if word not in stop_words])
    tmp = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(tmp)])
    tmp_bi = nltk.bigrams(nltk.word_tokenize(tmp))
    return FreqDist(tmp_bi)

def get_emolex(text):
    filepath = "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    emolex_dataframe = pd.read_csv(filepath,  names=["word", "emotion", "association"], sep='\t', keep_default_na=False)
    emotion_scores = emolex_dataframe.set_index(['emotion', 'word'])['association'].to_dict()

    # processed_text = preprocess_text(text)
    emotion_score = np.array([])
    # Filter out 'positive' and 'negative' emotions
    filtered_emotions = emolex_dataframe['emotion'].unique()[~np.isin(emolex_dataframe['emotion'].unique(), ['positive', 'negative'])]
    for e in filtered_emotions:
        score = np.mean([emotion_scores.get((e, wrd), 0) for wrd in text.split(" ")])
        # Store the average emotion scores in the dictionary
        emotion_score = np.append(emotion_score, score)
    return emotion_score

def load_lst():
    path = 'processed/frequency_distributions.pkl'
    with open(path, 'rb') as f:
        uni_lst = pickle.load(f)
        bi_lst = pickle.load(f)
    return uni_lst,bi_lst   

def unigram_features(tweet):
    uni_lst,bi_lst = load_lst()
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    unigram_freqs = []
    for i, emotion in enumerate(emotions):
        unigram_freq = sum([uni_lst[i].get(wrd) / len(uni_lst[i].keys()) if uni_lst[i].get(wrd) is not None else 0 for wrd in nltk.word_tokenize(tweet)])
        unigram_freqs.append(unigram_freq)

    return unigram_freqs

def bigram_features(tweet):
    uni_lst,bi_lst = load_lst()
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    bigram_freqs = []
    for i, emotion in enumerate(emotions):
        bigram_freq = sum([bi_lst[i].get(tpl) / len(bi_lst[i].keys()) if bi_lst[i].get(tpl) is not None else 0 for tpl in nltk.bigrams(nltk.word_tokenize(tweet))])
        bigram_freqs.append(bigram_freq)

    return bigram_freqs

def get_average_score(df):
    average_column = []
    # Iterate through each row of the DataFrame
    for i in range(len(df)):
        temp_arr = []
        for x in range(8):  # Assuming you want to calculate averages for the first 7 elements
            total = df['emotion_score'][i][x] + df['unigram_freq'][i][x] + df['bigram_freq'][i][x] + df['emolex'][i][x]
            average = total / 4
            temp_arr.append(average)
        # Convert the list of averages to a string and store it in the DataFrame
        average_column.append(temp_arr)
    df['average_score'] = average_column
    
    return df

def create_dataframe_for_tweet(clean_tweet):
    df_single = pd.DataFrame({'processed': [clean_tweet]})
    df_single['emotion_score'] = df_single['processed'].apply(get_emotion_score)
    df_single['unigram_freq'] = df_single['processed'].apply(unigram_features)
    df_single['bigram_freq'] = df_single['processed'].apply(bigram_features)
    df_single['emolex'] = df_single['processed'].apply(get_emolex)
    df_single = get_average_score(df_single)
    return df_single

def create_new_dataframe(df_single):
    new_df_single = pd.DataFrame()
    new_df_single['tweets'] = df_single['processed']
    new_df_single['average_score'] = df_single['emotion_score']
    return new_df_single
