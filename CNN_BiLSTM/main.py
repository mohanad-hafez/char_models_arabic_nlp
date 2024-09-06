import tensorflow as tf
import json
import os
import pandas as pd
from data_utils import Data
from models.char_cnn import CharCNN
from models.char_bilstm import CharBiLSTM
import numpy as np
import sys

tf.compat.v1.flags.DEFINE_string("model", "cnn", "Specifies which model to use: cnn or bilstm")
FLAGS = tf.compat.v1.flags.FLAGS

FLAGS(sys.argv)

model_name = FLAGS.model

project_dir = os.path.dirname(os.path.abspath(__file__))

data_source = os.path.join(project_dir, '..', 'data', 'saudi_privacy_policy')
alphabet= "ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآئءؤة,؟.!?:0123456789 "
input_size=3000
num_of_classes=10

# Load training data
training_data = Data(data_source=os.path.join(data_source, "train.csv"),
                     alphabet=alphabet,
                     input_size=input_size,
                     num_of_classes=num_of_classes)
training_data.load_data()
training_inputs, training_labels = training_data.get_all_data()

# Load validation data 
validation_data = Data(data_source=os.path.join(data_source, "val.csv"),
                       alphabet=alphabet,
                       input_size=input_size,
                       num_of_classes=num_of_classes)
validation_data.load_data()
validation_inputs, validation_labels = validation_data.get_all_data()

# Load test data
test_data = Data(data_source=os.path.join(data_source, "test.csv"),
                 alphabet=alphabet,
                 input_size=input_size,
                 num_of_classes=num_of_classes)
test_data.load_data()
test_inputs, test_labels = test_data.get_all_data()


# Build model
if model_name == "cnn":
    model = CharCNN(input_size=input_size,
                   alphabet_size=len(alphabet),
                   num_of_classes=num_of_classes)
    
elif model_name == "bilstm":
    model = CharBiLSTM(input_size=input_size,
                      alphabet_size=len(alphabet),
                      num_of_classes=num_of_classes)
    
# Train model
model.train(training_inputs=training_inputs,
            training_labels=training_labels,
            validation_inputs=validation_inputs,
            validation_labels=validation_labels)


# Test model
predictions, accuracy, loss = model.test(test_inputs, test_labels)

class_names = [str(i) for i in range(num_of_classes)]
model.analyze_results(predictions, test_inputs, test_labels, class_names)