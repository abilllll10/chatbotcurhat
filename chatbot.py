import io
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

#Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("halo")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """if user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

ADDITIONAL_RESPONSES = {
    "aku lagi sedih nih": "sedih kenapa? sini cerita",
    "aku disakiti pacarku": "emangnya pacarmu ngapain kamu?",
    "aku di selingkuhin sama dia": "wahh sudah jangan bersedih lagi kamu pasti bisa dapetin cowo yang lebih baik dari dia",
    
}

def response(user_response):
    """Generate response for user input"""
    robo_response = ''
    sent_tokens.append(user_response)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "Saya minta maaf, saya tidak mengerti."
    else:
        robo_response = sent_tokens[idx]
    
    # Remove the user response from sent_tokens to prevent duplicates
    sent_tokens.remove(user_response)
    
    # Check if the user input has an additional response
    if user_response.lower() in ADDITIONAL_RESPONSES:
        robo_response = ADDITIONAL_RESPONSES[user_response.lower()]
    
    return robo_response
