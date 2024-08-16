from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Bidirectional, LSTM,Convolution1D, GlobalMaxPooling1D, Flatten, Embedding, AlphaDropout
from keras.callbacks import  EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from keras import regularizers
from keras.optimizers import Adam
import keras
import os
import tensorflow as tf

# Setting random seeds to insure reproducibility
# Seed value
seed = 42

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only one GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()


def reset_seed(seed):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


class CharCNN_LSTM(object):
    def __init__(self, input_size, alphabet_size, num_of_classes):
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
        plt.title(f'Confusion Matrix - bilstm')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'figures/bilstm_confusion_matrix_cnn_bilstm.png')
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
        with open(f'reports/bilstm_classification_report_cnn_bilstm.txt', 'w') as f:
            f.write(f"Classification Report for cnn-bilstm\n\n")
            f.write(f"Micro-average F1 Score: {micro_f1:.4f}\n\n")
            f.write(report)

        print(f"Classification report saved to reports/cnn-bilstm_classification_report.txt")


    def _build_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        reset_seed(seed)
         # To control randomness
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, 15, embeddings_initializer=kernel_initializer)(inputs)

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

        # BiLSTM
        x = Bidirectional(LSTM(64, kernel_initializer=kernel_initializer, return_sequences=True), merge_mode='mul')(x)
        x = Flatten()(x)

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
        print("CharCNN_LSTM model built: ")
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

        early_stopping = EarlyStopping(monitor='val_loss',
                                              patience=7)

        model_checkpoint = ModelCheckpoint(
            f'best_model_bilstm_cnn_bilstm.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Start training
        print("Training CharCNN-BiLSTM model: ")
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
        plt.title(f'Model Accuracy - cnn-bilstm')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'figures/cnn-bilstm_accuracy_cnn_bilstm.png')
        plt.close()

        # plot loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss - cnn-bilstm')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'figures/cnn-bilstm_loss_cnn_bilstm.png')
        plt.close()

    def test(self, testing_inputs, testing_labels, batch_size, experiment_name):
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
        best_model_path = f'best_model_cnn-bilstm_cnn_bilstm.keras'
        self.model = keras.models.load_model(best_model_path)
        print(f"Loaded best model from {best_model_path}")

        # Evaluate inputs
        loss, accuracy = self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)

        # Get predictions
        predictions = self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)

        return predictions, loss, accuracy