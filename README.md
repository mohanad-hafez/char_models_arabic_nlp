
# Leveraging Character-Level Models and Data Augmentation to Enhance Arabic-NLP Tasks

This repository contains the source code for our research on character-level models in Arabic Natural Language Processing (NLP). The study addresses the challenges posed by Arabic's complex morphology, exploring various models, including Convolutional Neural Networks (CNNs), pre-trained transformers (CANINE), and Bidirectional Long Short-Term Memory networks (BiLSTMs). The research also investigates the impact of different data augmentation techniques, including a novel vowel deletion method, on the performance of these models in Arabic privacy policy classification.

## How to Run
### Prerequisites
Before running the code, make sure you have installed all the dependencies listed in the `requirements.txt` file:
```
pip install -r requirements.txt
```

### Dataset Preparation
The original dataset used in this study is not included in this repository. You can download it from https://github.com/iwan-rg/Saudi_Privacy_policy.

If you want to use the provided augmentation scripts or run the models, you’ll need to:
1. Download the dataset from the link above.
2. Split the dataset into train.csv, test.csv, and val.csv on your own (this is a straightforward process).
3. Place the split files into the ```data/saudi_privacy_policy folder```. The scripts expect the data to be in this location.

### Running the Models
#### CNN and BiLSTM Models
To run the CNN or BiLSTM models, use the appropriate flag in the `main.py` file within the `CNN_BiLSTM` directory:
```
cd CNN_BiLSTM
python main.py --model=cnn
python main.py --model=bilstm
```

#### CANINE Model
To run the CANINE model:
```
cd CANINE
python main.py
```

### Using Augmented Training Sets
The repository includes several augmented training sets within the `data/saudi_privacy_policy` folder. If you wish to use these datasets, update the data path in the `main.py` files to point to the appropriate augmented dataset.

### Generating Augmented Datasets
The scripts used to generate the augmented training sets are also included in this repository. If you wish to create new augmented datasets or modify the existing ones, you can run these scripts, which are located in the `augmentation` folder.

### Paper

https://aclanthology.org/2025.coling-main.186.pdf


### Cite
```
@inproceedings{mohamed2025enhancing,
  title={Enhancing Arabic NLP Tasks through Character-Level Models and Data Augmentation},
  author={Mohamed, Mohanad and Al-Azani, Sadam},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  pages={2744--2757},
  year={2025}
}
```


