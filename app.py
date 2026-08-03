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


def ask_ayurveda_ai(user_symptom: str, language: str):
    # Step A: Query Vector Database based on user symptom
    results = collection.query(query_texts=[user_symptom], n_results=2)
    retrieved_context = "\n\n".join(results["documents"][0])

    if not retrieved_context.strip():
        if language == "hi":
            return "मुझे इस समस्या के लिए डेटाबेस में कोई सटीक औषधि नहीं मिली।"
        return "Sorry, no relevant information was found in the database."

    # Step B: Instruct the LLM
    system_prompt = f"""
    You are an expert Ayurvedic Doctor strictly relying on the Bhavaprakasha Nighantu.
    A patient states their symptom: "{user_symptom}".
    
    Based ONLY on the following retrieved classical database records:
    ---
    {retrieved_context}
    ---
    
    Recommend the best matching herb. Keep it brief so it can be spoken aloud. 
    Format your response like this:
    **Sanskrit Name:** [Name]
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
    # Removes markdown asterisks (**) so the TTS doesn't say "asterisk asterisk" aloud
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