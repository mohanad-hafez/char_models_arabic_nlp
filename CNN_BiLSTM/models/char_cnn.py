from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Layer, BatchNormalization
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from keras import regularizers
from keras.optimizers import Adam
import keras
from keras.losses import CategoricalCrossentropy
import os
#from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback, ProgbarLogger
import tensorflow as tf



# Setting random seeds to insure reproducibility
# Seed value
seed = 42

def reset_seed(seed):
    keras.utils.set_random_seed(seed)


class CharCNN(object):
    """
    Class to implement the Character Level Convolutional Neural Network
    as described in Kim et al., 2015 (https://arxiv.org/abs/1508.06615)

    Their model has been adapted to perform text classification instead of language modelling
    by replacing subsequent recurrent layers with dense layer(s) to perform softmax over classes.
    """
    def __init__(self, input_size, alphabet_size,
                 num_of_classes):
        """
        Initialization for the Character Level CNN model.

        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            num_of_classes (int): Number of classes in data
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.num_of_classes = num_of_classes
        self._build_model()  # builds self.model variable
        

    def analyze_results(self, predictions, test_inputs, true_labels, class_names):
        # Convert predictions to class labels
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(true_labels, axis=1)
        
        # Find misclassified examples
        misclassified = np.where(pred_labels != true_labels)[0]
        print(f"Number of misclassified examples: {len(misclassified)}")

        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - cnn')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('figures/cnn_confusion_matrix.png')
        plt.close()

        # Calculate and print micro-average F1 score
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        print(f"\nMicro-average F1 Score: {micro_f1:.4f}")

        # Generate classification report
        report = classification_report(true_labels, pred_labels, target_names=class_names)
        print("\nClassification Report:")
        print(report)

        # Save classification report to a file
        os.makedirs('reports', exist_ok=True)
        with open('reports/cnn_classification_report.txt', 'w') as f:
            f.write("Classification Report for {experiment_name}\n\n")
            f.write("Micro-average F1 Score: {micro_f1:.4f}\n\n")
            f.write(report)

        print(f"Classification report saved to reports/cnn_classification_report.txt")


    def _build_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        reset_seed(seed)
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, 15)(inputs)
        # Convolution layers
        convolution_output = []
        conv1 = Convolution1D(filters=265,
                                 kernel_size=10,
                                 activation='relu',
                                 name='Conv1D_256_10',
                                 kernel_regularizer=regularizers.l2(0.01)
                                 )(x)
        
        pool1 = GlobalMaxPooling1D(name='MaxPoolingOverTime_256_10')(conv1)
        convolution_output.append(pool1)

        conv2 = Convolution1D(filters=265,
                                 kernel_size=7,
                                 activation='relu',
                                 name='Conv1D_256_7',
                                 kernel_regularizer=regularizers.l2(0.01)
                                 )(x)
        
        pool2 = GlobalMaxPooling1D(name='MaxPoolingOverTime_256_7')(conv2)
        convolution_output.append(pool2)

        conv3 = Convolution1D(filters=265,
                                 kernel_size=5,
                                 activation='relu',
                                 name='Conv1D_256_5',
                                 kernel_regularizer=regularizers.l2(0.01)
                                 )(x)
        
        pool3 = GlobalMaxPooling1D(name='MaxPoolingOverTime_256_5')(conv3)
        convolution_output.append(pool3)

        conv4 = Convolution1D(filters=265,
                                 kernel_size=3,
                                 activation='relu',
                                 name='Conv1D_256_3',
                                 kernel_regularizer=regularizers.l2(0.01)
                                 )(x)
        
        pool4 = GlobalMaxPooling1D(name='MaxPoolingOverTime_256_3')(conv4)
        convolution_output.append(pool4)

        x = Concatenate()(convolution_output)
        
        # Dense layer
        x = Dense(512, activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(x)
        x = AlphaDropout(0.25)(x)

        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)

        # Build and compile model
        model = Model(inputs=inputs, outputs=predictions)

        # Optimizer
        adam = Adam(learning_rate = 0.0001)

        # Loss function
        loss = 'categorical_crossentropy'

        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

        self.model = model
        print("CharCNN model built: ")
        self.model.summary()
        

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels):
        """
        Training function

        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels

        Returns: None

        """
        reset_seed(seed)
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                              patience=7)

        model_checkpoint = ModelCheckpoint(
            f'best_model_cnn.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Start training
        print("Training CharCNNKim model: ")
        history = self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=300,
                       batch_size=128,
                       verbose=1,
                       callbacks=[model_checkpoint]
                       )
        # plot accuracy
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy - cnn')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'figures/cnn_accuracy.png')
        plt.close()

        # plot loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss - cnn')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'figures/cnn_loss.png')
        plt.close()

    def test(self, testing_inputs, testing_labels):
        """
        Testing function

        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        reset_seed(seed)

        # Load the best model
        best_model_path = f'best_model_cnn.keras'
        self.model = keras.models.load_model(best_model_path)
        print(f"Loaded best model from {best_model_path}")

        # Evaluate inputs
        loss, accuracy = self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)

        # Get predictions
        predictions = self.model.predict(testing_inputs, batch_size=128, verbose=1)

        return predictions, loss, accuracy
