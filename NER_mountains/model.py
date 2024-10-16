from tensorflow.keras.models import Sequential
from pathlib import Path
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
import matplotlib.pyplot as plt



class MountainNERModel:
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_tokenizer = None
        self.tag2idx = None
        self.idx2tag = None
        self.max_length = None
        self.model = None
        self.vocab_size = None
        self.num_tags = None

    def prepare_sequences(self, sentences):

        # Create word tokenizer if not exists
        if self.word_tokenizer is None:
            self.word_tokenizer = Tokenizer()
            self.word_tokenizer.fit_on_texts(sentences)
            self.vocab_size = len(self.word_tokenizer.word_index) + 1

        # Set max length if not exists
        if self.max_length is None:
            self.max_length = max(len(s.split()) for s in sentences)

        sequences = self.word_tokenizer.texts_to_sequences(sentences)
        return pad_sequences(sequences, maxlen=self.max_length, padding='post')

    def prepare_tags(self, tags):
        # Create tag dictionary if not exists
        if self.tag2idx is None:
            unique_tags = set([tag for sequence in tags for tag in sequence.split()])
            self.tag2idx = {'I-MOUNTAIN': 1, 'O': 0, 'B-MOUNTAIN': 2}
            self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
            self.num_tags = len(self.tag2idx)

        seq_tags = [[self.tag2idx[tag] for tag in sequence.split()] for sequence in tags]
        pad_tags=pad_sequences(seq_tags, maxlen=self.max_length, padding='post')
        return to_categorical(pad_tags, num_classes=len(self.tag2idx))

    def create_model(self):
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Dropout(0.2),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            TimeDistributed(Dense(self.num_tags, activation='softmax'))
        ])
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    def fit_model(
        self,
        sentence_train,
        tag_train,
        sentence_val,
        tag_val,
        epochs=10
    ):
        history = self.model.fit(
            sentence_train, tag_train,
            validation_data=(sentence_val, tag_val),
            epochs=epochs,
            batch_size=32
        )
        return history





    def plot_training_history(self, history):
        """
        Plot training history metrics
        """
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Function to evaluate the model
    def evaluate_model(self, X, y):
        return self.model.evaluate(X, y)

    def save_model(self) -> None:
        # Save the model
        model_folder = Path(__file__).resolve().parent.parent / "model"
        model_folder.mkdir(parents=True, exist_ok=True)
        model_path = model_folder / "model.h5"
        self.model.save(str(model_path))















