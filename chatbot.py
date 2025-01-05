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
GREETING_RESPONSES = ["hi", "hey", "hadir!", "hello", "curhat dong disini!"]

def greeting(sentence):
    """if user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

ADDITIONAL_RESPONSES = {
    "aku lagi sedih nih": "sedih kenapa? sini cerita",
    "aku disakiti pacarku": "emangnya pacarmu ngapain kamu?",
    "saya ragu memilih jurusan kuliah": "Pertimbangkan minat dan bakat",
    "aku harus gimana": "kamu harus melakukan hal-hal yang kamu suka pasti nanti kamu akan lupa dengannya",
    "aku mau cerita": "silahkan, mau cerita apa?",
    "aku mau curhat": "silahkan, mau curhat apa?",
    "saya merasa tidak berguna": "Anda berguna! Fokus pada kebaikan Anda",
    "bagaimana cara menghadapi kesedihan?": "Terima perasaan, ceritakanlah, dan cari dukungan",
    "apa yang membuat hidup berarti?": "Menemukan tujuan dan menjalani hidup dengan penuh makna",
    "saya sedang merasa marah": "Ambil napas dalam-dalam, tunggu sejenak sebelum bertindak",
    "kenapa ya temanku akhir-akhir ini jadi cuek?": "Mungkin dia lagi ada masalah sendiri. Coba tanya, siapa tahu dia butuh dukunganmu",
    "gimana caranya menghargai diri sendiri?": "Mulai dari hal kecil, seperti memuji dirimu atas hal baik yang sudah kamu lakukan",
    "apa yang bikin hidup ini terasa berat?": "Kadang ekspektasi kita sendiri yang bikin berat. Coba kurangi tekanan ke diri sendiri",
    "apa yang membuat hidup indah?": "Kebahagiaan, cinta, dan hubungan yang baik",
    "bagaimana cara mengatasi kecemasan?": "Coba teknik pernapasan atau konsultasi profesional",
    "saya merasa kesepian": "Hubungi teman atau keluarga, atau coba hobi baru",
    "kenapa ya aku sering overthinking tentang masa depan?": "Mungkin kamu terlalu fokus ke hal yang belum terjadi. Coba lebih menikmati momen sekarang",
    "kenapa ya aku masih kepikiran kesalahan yang udah lama terjadi?": "Itu normal. Tapi, coba lihat kesalahan itu sebagai pelajaran, bukan beban",
    "aku sering nggak cocok sama pasangan, gimana dong?": "Mungkin kamu butuh komunikasi lebih terbuka. Coba deh obrolin perasaanmu",
    "kenapa ya aku susah percaya sama orang baru?": "Mungkin kamu pernah dikecewakan. Pelan-pelan aja, kepercayaan itu dibangun seiring waktu",
    "gimana caranya tahu kalau dia beneran tulus sama aku?": "Lihat dari tindakannya, bukan hanya kata-katanya. Tindakan biasanya nggak bohong",
    "gimana cara menghadapi orang yang toxic?": "Jaga jarak dan fokus pada kesehatan mentalmu. Nggak apa-apa untuk memilih lingkungan yang lebih baik",
    "apa aku harus minta maaf meskipun aku nggak salah?": "Kadang meminta maaf bisa meredakan konflik, tapi pastikan itu nggak mengorbankan harga dirimu",
    "kenapa ya aku merasa hidupku nggak berubah dari dulu?": "Mungkin kamu perlu mencoba hal baru atau keluar dari zona nyamanmu",
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
