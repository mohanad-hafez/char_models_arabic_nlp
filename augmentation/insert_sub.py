import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm.auto import tqdm
import os
import unicodedata
import re


action='insert' # 'insert' or 'substitute'

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

def augment_arabic_dataset(df):
  aug = naw.ContextualWordEmbsAug(
      model_path='aubmindlab/bert-base-arabertv02',
      device='cuda',
      action=action) 
  # Create a progress bar using tqdm
  tqdm.pandas(desc="Augmenting", ncols=80, dynamic_ncols=True) # Apply tqdm to Pandas
  def augment_row(row):
      original_text = row['text']
      augmented_text = aug.augment(original_text)[0]
      augmented_text = normalize(augmented_text) # applying the same normalization they applied after augmentation
      augmented_text = remove_diacritics(augmented_text)
      return pd.Series({'label': row['label'], 'text': augmented_text})

  augmented_df = df.progress_apply(augment_row, axis=1)
  return pd.concat([df, augmented_df], ignore_index=True)

project_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_dir,"..", "data", "saudi_privacy_policy", "train.csv")

print("reading the data")
df = pd.read_csv(data_path, header=None, names=['label', 'text'])
print("augmenting...")
augmented_df = augment_arabic_dataset(df)
print("completed.")
shuffled_df = augmented_df.sample(frac=1, random_state=42)

output_path = os.path.join(project_dir,"..", "data", "Saudipp", f"train_{action}.csv")
shuffled_df.to_csv(output_path, index=False, header=False)

print("original train size:", len(df))
print("augmented train size:", len(shuffled_df))