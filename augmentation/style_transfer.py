import os
import weave
from openai import OpenAI
from tqdm.auto import tqdm
import re
import unicodedata
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''

def remove_diacritics(text):
    # Normalize the text to decompose combined characters
    normalized = unicodedata.normalize('NFD', text)
    
    # Remove all diacritical marks
    without_diacritics = ''.join([c for c in normalized if not unicodedata.combining(c)])
    
    # Remove tatweel (elongation character)
    without_tatweel = re.sub('[ـ]', '', without_diacritics)
    
    return without_tatweel


def normalize(text):
    replacements = {
        'إ': 'ا', 'أ': 'ا', 'ٱ': 'ا', 'آ': 'ا',
        'ؤ': 'و',
        'ئ': 'ي', 'ى': 'ي',
        'ة': 'ه'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text



# Initialize Weave and W&B
weave.init('gpt-4o-mini-aug')


# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = input("Enter openAI API key:")


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@weave.op()
def augment(text):
    gpt_user_prompt = f"""
    Please rewrite the following Arabic text, changing every nominal sentence to verbal and every verbal sentence to nominal. Maintain the original meaning as closely as possible. Provide only the transformed Arabic text.
    Sentence: {text}
    """

    messages = [
        {"role": "user", "content": gpt_user_prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        frequency_penalty=0.0,
        stop=None
    )
    response_text = response.choices[0].message.content
    #print(response_text)
    #tokens_used = response.usage.total_tokens
    
    return response_text



print("beginning...")
def augment_arabic_dataset(df):

    # Create a progress bar using tqdm
    tqdm.pandas(desc="Augmenting", ncols=80, dynamic_ncols=True) # Apply tqdm to Pandas
    def augment_row(row):
        original_text = row['text']
        original_text=original_text[:2000]
        augmented_text = augment(original_text)
        augmented_text = normalize(augmented_text)
        augmented_text = remove_diacritics(augmented_text)
        return pd.Series({'label': row['label'], 'text': augmented_text})

    augmented_df = df.progress_apply(augment_row, axis=1)
    return pd.concat([df, augmented_df], ignore_index=True)


# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the data path using os.path.join
data_path = os.path.join(project_dir, "..", "data", "saudi_privacy_policy", "train.csv")
output_path = os.path.join(project_dir,"..",  "data", "saudi_privacy_policy", "train_back.csv")


df = pd.read_csv(data_path, header=None, names=['label', 'text'])
df = df.sample(3)
augmented_df = augment_arabic_dataset(df)
shuffled_df = augmented_df.sample(frac=1, random_state=42)
shuffled_df.to_csv(output_path, index=False, header=False)

print("original train size:", len(df))
print("augmented train size:", len(shuffled_df))
