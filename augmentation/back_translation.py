import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm.auto import tqdm  # Import tqdm for progress bars
import os
import re
import unicodedata

os.environ['CURL_CA_BUNDLE'] = ''


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


def remove_diacritics(text):
    # Normalize the text to decompose combined characters
    normalized = unicodedata.normalize('NFD', text)

    # Remove all diacritical marks
    without_diacritics = ''.join([c for c in normalized if not unicodedata.combining(c)])

    # Remove tatweel (elongation character)
    without_tatweel = re.sub('[ـ]', '', without_diacritics)

    return without_tatweel


def augment_arabic_dataset(df):

    back_translation_aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-ar-en',
        to_model_name='Helsinki-NLP/opus-mt-en-ar',
        device='cuda'
    )
    # Create a progress bar using tqdm
    tqdm.pandas(desc="Augmenting", ncols=80, dynamic_ncols=True)  # Apply tqdm to Pandas

    def augment_row(row):
        original_text = row['text']
        original_text = original_text[:2000]
        augmented_text = back_translation_aug.augment(original_text)[0]
        augmented_text = normalize(augmented_text)  # applying the same normalization they applied after augmentation
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
augmented_df = augment_arabic_dataset(df)
shuffled_df = augmented_df.sample(frac=1, random_state=42)
shuffled_df.to_csv(output_path, index=False, header=False)

print("original train size:", len(df))
print("augmented train size:", len(shuffled_df))