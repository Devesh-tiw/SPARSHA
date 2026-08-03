import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLAMA_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Load your master file
df = pd.read_csv("Master_Ayurveda_Database_Fixed.csv")

print(f"Total rows to process: {len(df)}")

for index, row in df.iterrows():
    # Only translate if english_symptoms is currently empty
    if pd.isna(row.get('english_symptoms')) or str(row.get('english_symptoms')).strip() == "":
        karma_text = str(row.get('raw_karma_sanskrit', '')).strip()
        
        # Skip if there's no karma to translate
        if not karma_text or karma_text == "nan":
            continue
            
        prompt = f"""
        Translate the following Ayurvedic Sanskrit symptoms/actions into modern English medical terms. 
        Return ONLY a raw JSON object with the key "english" and the string value. Do not include markdown formatting.
        Sanskrit: {karma_text}
        """
        
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            # Remove any stray markdown
            if result_text.startswith("```json"): result_text = result_text[7:]
            if result_text.endswith("```"): result_text = result_text[:-3]
            
            data = json.loads(result_text)
            df.at[index, 'english_symptoms'] = data.get("english", "")
            
            print(f"Row {index} translated: {data.get('english', '')[:50]}...")
            
            # Save every 5 rows to prevent data loss
            if index % 5 == 0:
                df.to_csv("Master_Ayurveda_Database_Fixed.csv", index=False, encoding='utf-8-sig')
                
            time.sleep(1) # Be nice to the API rate limits
            
        except Exception as e:
            print(f"Error on row {index}: {e}")

# Final save
df.to_csv("Master_Ayurveda_Database_Fixed.csv", index=False, encoding='utf-8-sig')
print("✅ Translation complete!")