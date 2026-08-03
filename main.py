import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

if not LLAMA_API_KEY:
    print("CRITICAL ERROR: LLAMA_API_KEY not found in .env file!")
    exit()

# OpenRouter Client Setup
client = OpenAI(
    api_key=LLAMA_API_KEY,
    base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./ayurveda_vector_db")
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="bhavprakash_collection", embedding_function=sentence_transformer_ef
)


def ask_ayurveda_ai(user_symptom: str, language: str = "en"):
    # Step A: Query ChromaDB
    results = collection.query(query_texts=[user_symptom], n_results=2)

    retrieved_context = "\n\n".join(results["documents"][0])

    if not retrieved_context.strip():
        return "Sorry, no relevant information was found in the database."

    # Step B: Strict System Prompt
    system_prompt = f"""
    You are an expert Ayurvedic Doctor strictly relying on the Bhavaprakasha Nighantu.
    A patient states their symptom: "{user_symptom}".
    
    Based ONLY on the following retrieved classical database records:
    ---
    {retrieved_context}
    ---
    
    Recommend the appropriate herbs. You MUST format your response exactly like this for each herb:

    **Sanskrit Name:** [Herb Hindi/Sanskrit Name]
    **English/Botanical Name:** [English and Botanical name]
    **Category (Varga):** [Varga Name]
    
    **Classical Properties:**
    * **Rasa (Taste):** [Extract from text]
    * **Guna (Qualities):** [Extract from text]
    * **Virya (Potency):** [Extract from text]
    * **Vipaka (Post-digestive):** [Extract from text]
    
    **Clinical Action (Karma):**
    [List the raw Sanskrit karma and its translation]
    """

    # Step C: Call OpenRouter Model
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",  # OpenRouter model ID
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"I need a remedy for: {user_symptom}",
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error connecting to OpenRouter API: {e}"


if __name__ == "__main__":
    user_input = "I have a severe cough and chest congestion."
    print("User:", user_input)
    print("\n--- AI RESPONSE ---")
    print(ask_ayurveda_ai(user_input, language="en"))