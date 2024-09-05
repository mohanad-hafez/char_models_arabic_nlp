from keras.models import Model
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, Add, Flatten
from keras.layers import AlphaDropout, BatchNormalization
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
from keras import regularizers
import tensorflow as tf
import os




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

class CharBiLSTM(object):
    def __init__(self, input_size, alphabet_size,
                 num_of_classes):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.num_of_classes = num_of_classes
        self._build_model()

    def analyze_results (self, predictions, test_inputs, true_labels, class_names):
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
        plt.title('Confusion Matrix - bilstm')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('figures/bilstm_confusion_matrix_cnn_bilstm.png')
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
        with open('reports/bilstm_classification_report_cnn_bilstm.txt', 'w') as f:
            f.write("Classification Report for bilstm\n\n")
            f.write("Micro-average F1 Score: {micro_f1:.4f}\n\n")
            f.write(report)

        print(f"Classification report saved to reports/bilstm_classification_report.txt")

    def _build_model(self):
        reset_seed(seed)

        # To control randomness
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')

        # Embedding layer
        x = Embedding(self.alphabet_size + 1, 15, embeddings_initializer=kernel_initializer)(inputs)

        # Bidirectional layer
        x = Bidirectional(LSTM(128, kernel_initializer=kernel_initializer, return_sequences=True), merge_mode='mul')(x)
        x = Flatten()(x)
        
        # Dense layer
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=kernel_initializer)(x)
        x = AlphaDropout(0.15)(x)
        
        predictions = Dense(self.num_of_classes, activation='softmax', kernel_initializer=kernel_initializer)(x)
        model = Model(inputs=inputs, outputs=predictions)
        
        adam = Adam(learning_rate=0.001)
        loss = 'categorical_crossentropy'
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        self.model = model
        print("CharBiLSTM model built: ")
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
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard

        Returns: None

        """
        reset_seed(seed)

        # Create callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                              patience=7)

        model_checkpoint = ModelCheckpoint(
            f'best_model_bilstm.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        # Start training
        print("Training CharBiLSTM model: ")
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
        plt.title('Model Accuracy - bilstm')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('figures/bilstm_accuracy_bilstm.png')
        plt.close()

        # plot loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss - bilstm')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('figures/bilstm_loss_bilstm.png')
        plt.close()

    def test(self, testing_inputs, testing_labels, batch_size):
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
        best_model_path = f'best_model_bilstm_bilstm.keras'
        self.model = keras.models.load_model(best_model_path)
        print(f"Loaded best model from {best_model_path}")

        # Evaluate inputs
        loss, accuracy = self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)

        # Get predictions
        predictions = self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)

        return predictions, loss, accuracy