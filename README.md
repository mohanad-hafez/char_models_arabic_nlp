
# Leveraging Character-Level Models and Data Augmentation to Enhance Arabic-NLP Tasks

This repository contains the source code for our research on character-level models in Arabic Natural Language Processing (NLP). The study addresses the challenges posed by Arabic's complex morphology, exploring various models, including Convolutional Neural Networks (CNNs), pre-trained transformers (CANINE), and Bidirectional Long Short-Term Memory networks (BiLSTMs). The research also investigates the impact of different data augmentation techniques, including a novel vowel deletion method, on the performance of these models in Arabic privacy policy classification.

## How to Run
### Prerequisites
Before running the code, make sure you have installed all the dependencies listed in the `requirements.txt` file:
```
pip install -r requirements.txt
```
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

### Cite


