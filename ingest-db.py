import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
PERSIST_PATH = "./chroma_ayurveda_db"   # change this if your DB lives elsewhere

client = chromadb.PersistentClient(path=PERSIST_PATH)
collections = client.list_collections()

if not collections:
    print(f"No collections found at {PERSIST_PATH}. Wrong path, or DB was never persisted here.")
else:
    for c in collections:
        col = client.get_collection(c.name)
        print(f"Collection: {c.name}")
        print(f"  Document count: {col.count()}")
        print(f"  Metadata: {c.metadata}")
        if col.count() > 0:
            sample = col.peek(1)
            print(f"  Sample doc: {sample['documents'][0][:200]}")
            print(f"  Sample metadata: {sample['metadatas'][0]}")
# Set this to your exact master CSV filename
CSV_FILE_PATH = "Master_Ayurveda_Database_Fixed.csv"
DB_PATH = "./ayurveda_vector_db"

def ingest_data():
    # 1. Verify file existence
    if not os.path.exists(CSV_FILE_PATH):
        # Fallback check if the file is named without _Fixed
        if os.path.exists("Master_Ayurveda_Database.csv"):
            filepath = "Master_Ayurveda_Database.csv"
        else:
            print(f"❌ Error: Could not find '{CSV_FILE_PATH}' in current directory.")
            return
    else:
        filepath = CSV_FILE_PATH

    print(f"🚀 Initializing embedding model for {filepath}...")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    print(f"📁 Connecting to ChromaDB at {DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    collection = chroma_client.get_or_create_collection(
        name="bhavprakash_collection",
        embedding_function=embedding_fn
    )

    print(f"📖 Reading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"⏳ Ingesting {len(df)} records into ChromaDB. Please wait...")
    
    documents = []
    ids = []
    
    for index, row in df.iterrows():
        herb_hi = str(row.get('herb_hindi', '')).strip()
        varga = str(row.get('varga_category', '')).strip()
        synonyms = str(row.get('synonyms', '')).strip()
        props = str(row.get('classical_properties', '')).strip()
        karma = str(row.get('raw_karma_sanskrit', '')).strip()
        english_symp = str(row.get('english_symptoms', '')).strip() # <--- NEW LINE
        herb_en = str(row.get('herb_english', '')).strip()
        botanical = str(row.get('botanical_name', '')).strip()
        
        # Skip empty strings
        herb_hi = "" if herb_hi == "nan" else herb_hi
        synonyms = "" if synonyms == "nan" else synonyms
        props = "" if props == "nan" else props
        karma = "" if karma == "nan" else karma
        english_symp = "" if english_symp == "nan" else english_symp # <--- NEW LINE
        herb_en = "" if herb_en == "nan" else herb_en
        botanical = "" if botanical == "nan" else botanical

        # THE FIX: Now ChromaDB will actually index the English translated symptoms!
        text_content = (
            f"Herb Sanskrit/Hindi: {herb_hi} | "
            f"Varga Category: {varga} | "
            f"Synonyms: {synonyms} | "
            f"Classical Properties: {props} | "
            f"Karma/Indications (Sanskrit): {karma} | "
            f"Symptoms/Uses (English): {english_symp} | "
            f"English Name: {herb_en} | "
            f"Botanical Name: {botanical}"
        )
        
        documents.append(text_content)
        ids.append(f"bpn_{index}")

    # Batch add to vector database
    collection.add(
        documents=documents,
        ids=ids
    )

    print("✅ Successfully ingested all 1,110 records into ChromaDB!")

if __name__ == "__main__":
    ingest_data()