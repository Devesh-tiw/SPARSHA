from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import csv
import os
import time

app = Flask(__name__)


# Ensure the audio folder exists
os.makedirs('static/audio', exist_ok=True)

def get_ayurvedic_advice(user_input, lang_pref):
    search_query = user_input.lower().replace(",", "").replace("?", "").replace("।", "")
    # 1. Ignore common English words so they don't break the search
    stopwords = ["i", "have", "a", "am", "the", "my", "is", "and", "of", "lot", "pain", "in", "for", "only"]
    words_in_query = [w for w in search_query.split() if w not in stopwords]
    
    # 2. Choose Database and Translation
    if lang_pref == 'hi':
        symptom_dictionary = {
            "cough": "कफ", "cold": "जुकाम", "diarrhea": "अतीसार",
            "fever": "ज्वर", "digestion": "अग्निमान्द्य", "diabetes": "प्रमेह",
            "pitta": "पित्त", "vata": "वात", "kapha": "कफ",
            "mango": "आम्र", "headache": "शिरःशूल", "blood": "रक्तपित्त",
            "bukhar": "ज्वर"
        }
        
        search_words = [symptom_dictionary.get(word, word) for word in words_in_query]
        db_file = 'total_ayurveda_database.csv'
    else:
        search_words = words_in_query
        db_file = 'english_ayurveda_database.csv'
            
    try:
        best_match_herb = ""
        best_match_karma = ""
        highest_score = 0
        
        with open(db_file, mode='r', encoding='utf-8', errors='replace') as file:
            reader = csv.DictReader(file)
            current_herb = "Unknown Herb"
            
            for row in reader:
                karma = ""
                for key, value in row.items():
                    val_str = str(value).strip()
                    # FORCE the AI to only remember herbs that actually have a real name typed in the box!
                    if key and 'Head' in key and val_str and val_str != "":
                        current_herb = val_str
                    if key and 'Karma' in key:
                        karma = str(value).strip()
                
                row_text = " ".join([str(val).lower() for val in row.values() if val])
                
                # 3. The Advanced Scoring System
                score = 0
                for word in search_words:
                    if word and word.lower() in row_text:
                        score += 1
                        # BIG BONUS: If the symptom is specifically in the Karma (properties), give it 2 extra points!
                        if karma and word.lower() in karma.lower():
                            score += 2
                        
                # 4. Only update the winner if it has a HIGHER score AND a valid name!
                if score > highest_score and current_herb != "Unknown Herb" and current_herb != "":
                    highest_score = score
                    best_match_herb = current_herb
                    best_match_karma = karma
                    
        # 5. Return the Ultimate Winner
        if highest_score > 0:
            if lang_pref == 'en':
                return f"According to the database, the best herb is {best_match_herb}. Its properties are: {best_match_karma}.", 'en'
            else:
                return f"डेटाबेस के अनुसार, सबसे अच्छी औषधि '{best_match_herb}' है। इसके गुण हैं: {best_match_karma}।", 'hi'
        else:
            if lang_pref == 'en':
                return "I could not find a specific medicine for that in the database.", 'en'
            else:
                return "मुझे इस समस्या के लिए डेटाबेस में कोई सटीक औषधि नहीं मिली।", 'hi'
                
    except FileNotFoundError:
        return f"Error: The {db_file} file is missing.", lang_pref
# --- WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_ai():
    user_message = request.json.get("message")
    user_lang = request.json.get("language", "en") # Get the language from the website
    
    # 1. Get Answer
    answer_text, lang_code = get_ayurvedic_advice(user_message, user_lang)
    
    # 2. Generate Audio 
    audio_filename = f"response_{int(time.time())}.mp3"
    audio_path = os.path.join('static', 'audio', audio_filename)
    
    if lang_code == 'en':
        tts = gTTS(text=answer_text, lang='en', tld='co.in')
    else:
        tts = gTTS(text=answer_text, lang='hi')
        
    tts.save(audio_path)
    
    # 3. Send back to website
    return jsonify({
        "answer": answer_text,
        "audio_url": f"/{audio_path}"
    })

if __name__ == '__main__':
    app.run(debug=True)