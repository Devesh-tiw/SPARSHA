from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import os
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = Flask(__name__)


os.makedirs('static/audio', exist_ok=True)


load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLAMA_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./ayurveda_vector_db")
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="bhavprakash_collection", embedding_function=sentence_transformer_ef
)
def get_smart_search_query(raw_input: str) -> str:
    """
    Dynamically translates ANY user input into the perfect English + Sanskrit search terms.
    Works for all 1100+ diseases without needing if/else blocks.
    """
    prompt = f"""
    A patient stated their symptom (which may be in Hinglish or have typos): "{raw_input}"
    
    1. Identify the core disease or symptom.
    2. Translate it into standard English medical terms.
    3. Add the classical Ayurvedic Sanskrit name for that disease (e.g., Jwara for fever, Kamala for jaundice, Atisara for diarrhea, Shwasa for asthma, Kushtha for skin disease, etc.).
    
    Output ONLY a space-separated string of these highly relevant keywords. No punctuation, no sentences.
    
    Examples:
    - "mujhe bhkhar hai" -> fever jwara
    - "pet me dard" -> stomach ache abdominal pain shula udararoga
    - "peeliya ho gaya" -> jaundice kamala
    - "khasi aa rahi hai" -> cough kasa
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, 
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return raw_input.lower()


def ask_ayurveda_ai(user_symptom: str, language: str):
    
    search_query = get_smart_search_query(user_symptom)
    print(f"\nUser typed: {user_symptom} | AI Search Query: {search_query}")
    
    results = collection.query(query_texts=[search_query], n_results=5)
    retrieved_context = "\n\n".join(results["documents"][0])
    
  

    
    system_prompt = f"""
    You are an expert Ayurvedic Doctor strictly relying on the Bhavaprakasha Nighantu.
    A patient states their symptom: "{user_symptom}".
    
    Based ONLY on the following retrieved classical database records:
    ---
    {retrieved_context}
    ---
    
    CLINICAL RULES:
    1. IGNORE JUNK: Ignore chapter headings (like "अथ गव्यघृतस्य"), words like 'Madhyam', or blank rows.
    2. THE FEVER RULE: If the symptom is fever, do NOT recommend Ghee (Ghrita). Look for valid herbs in the context.
    3. MISSING DATA: If the English or Botanical name is missing in the retrieved text, simply omit them. Do not print empty slashes.
    
    Recommend the best matching herb. Keep it brief so it can be spoken aloud. 
    Format your response EXACTLY like this:
    **Herb Name:** [Sanskrit Name] (Include / English / Botanical ONLY if found in the text)
    **Properties:** [Rasa, Guna, Virya, Vipaka]
    **Uses:** [Brief translation of Karma]
    """

    if language == "hi":
        system_prompt += "\n\nCRITICAL: You MUST write your entire response in Hindi."

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"I need a remedy for: {user_symptom}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"
def clean_text_for_speech(text):
    
    return re.sub(r'\*+', '', text)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_ai():
    user_message = request.json.get("message")
    user_lang = request.json.get("language", "en") # 'en' or 'hi'
    
    
    answer_text = ask_ayurveda_ai(user_message, user_lang)
    
     
    clean_audio_text = clean_text_for_speech(answer_text)
    audio_filename = f"response_{int(time.time())}.mp3"
    audio_path = os.path.join('static', 'audio', audio_filename)
    
    try:
        if user_lang == 'en':
            tts = gTTS(text=clean_audio_text, lang='en', tld='co.in')
        else:
            tts = gTTS(text=clean_audio_text, lang='hi')
        tts.save(audio_path)
    except Exception as e:
        print("Audio generation failed:", e)
    
    
    return jsonify({
        "answer": answer_text,
        "audio_url": f"/{audio_path}"
    })

if __name__ == '__main__':
    app.run(debug=True)