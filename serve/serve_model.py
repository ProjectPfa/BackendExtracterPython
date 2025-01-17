import cv2
import numpy as np
import easyocr
import re
from difflib import SequenceMatcher

def read_image_file(file_content):
    np_array = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def preprocess_image(image):
    # Redimensionner avec une plus grande résolution
    height = 1200
    aspect_ratio = height / image.shape[0]
    width = int(image.shape[1] * aspect_ratio)
    image = cv2.resize(image, (width, height))
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Réduire le bruit tout en préservant les bords
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Augmenter la netteté
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def is_header_word(word, header_words):
    for header in header_words:
        if SequenceMatcher(None, word, header).ratio() > 0.7:
            return True
    return False

def extract_info(text):
   info = {
       'nom': '',
       'prenom': '',
       'date_naissance': '',
       'numero_id': ''
   }
   
   lines = text.split()
   header_words = {'ROYAUME','DTDENTIE','NANIONALE', 'MAROC', 'CARTE', 'NATIONALE', 'IDENTITE','RONAUME','CRI', 'DU'}
   
   # Trouver l'index après l'en-tête
   start_index = 0
   for i, word in enumerate(lines):
       if 'IDENTITE' in word:
           start_index = i + 1
           break
           
   # Chercher les premiers mots en majuscules après l'en-tête
   valid_words = []
   for word in lines[start_index:]:
       clean_word = ''.join(c for c in word if c.isascii() and c.isalpha()).upper()
       if (clean_word and len(clean_word) > 1 and 
           clean_word not in header_words and 
           all(c.isupper() for c in clean_word)):
           valid_words.append(clean_word)
           
   # Les deux premiers mots valides sont prénom et nom
   if len(valid_words) >= 2:
       info['prenom'] = valid_words[0]
       info['nom'] = valid_words[1]
       
   # Date et numéro ID
   for word in lines:
       if re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', word):
           info['date_naissance'] = word
       if re.match(r'[A-Z]\d+', word):
           info['numero_id'] = word
           
   return info

def predict(image):
    preprocessed = preprocess_image(image)
    
    # Configurer EasyOCR avec plus de paramètres
    reader = easyocr.Reader(['en', 'ar'], gpu=False)
    results = reader.readtext(
        preprocessed,
        paragraph=False,
        height_ths=0.8,
        width_ths=0.8,
        contrast_ths=0.1
    )
    
    text = ' '.join([text[1] for text in results])
    print("Texte extrait:", text)
    
    return extract_info(text)
