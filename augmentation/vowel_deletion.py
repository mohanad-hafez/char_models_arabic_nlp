import pandas as pd
import random
import numpy as np
import os

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def remove_random_vowel(word):
    vowels = 'اويى'
    vowel_positions = [i for i, char in enumerate(word) if char in vowels]
    if vowel_positions:
        pos_to_remove = random.choice(vowel_positions)
        return word[:pos_to_remove] + word[pos_to_remove + 1:]
    return word

def remove_first_vowel(word):
    vowels = 'اويى'
    vowel_positions = [i for i, char in enumerate(word) if char in vowels]
    if vowel_positions:
        pos_to_remove = vowel_positions[0]
        return word[:pos_to_remove] + word[pos_to_remove + 1:]
    return word

def remove_last_vowel(word):
    vowels = 'اويى'
    vowel_positions = [i for i, char in enumerate(word) if char in vowels]
    if vowel_positions:
        pos_to_remove = vowel_positions[-1]
        return word[:pos_to_remove] + word[pos_to_remove + 1:]
    return word

def augment_text(text, probability, augmentation_type):
    words = text.split()
    augmented_words = []
    for word in words:
        if len(word) > 4 and random.random() < probability:
            if augmentation_type == 'random':
                augmented_word = remove_random_vowel(word)
            elif augmentation_type == 'first':
                augmented_word = remove_first_vowel(word)
            elif augmentation_type == 'last':
                augmented_word = remove_last_vowel(word)
            augmented_words.append(augmented_word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the input and output paths using os.path.join
input_file = os.path.join(project_dir, "..", "data", "saudi_privacy_policy", "train.csv")

augmentation_types = ['random', 'first', 'last']

for aug_type in augmentation_types:
    df = pd.read_csv(input_file, header=None, names=['label', 'text'])
    augmented_df = df.copy()
    augmented_df['text'] = augmented_df['text'].apply(lambda x: augment_text(x, probability=0.3, augmentation_type=aug_type))
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    output_file = os.path.join(project_dir, "..", "data", "Saudipp", f"train_{aug_type}_vowel.csv")
    combined_df.to_csv(output_file, index=False, header=False)

    print(f"{aug_type} vowel augmentation:")
    print(f"Before augmentation: {len(df)}")
    print(f"After augmentation: {len(combined_df)}")
    print(f"Saved to: {output_file}\n")