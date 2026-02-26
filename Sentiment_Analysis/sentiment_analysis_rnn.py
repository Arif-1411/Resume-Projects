"""
============================================================
SENTIMENT ANALYSIS USING RNN & NLP
============================================================
Author: [Your Name]
Technologies: Python, TensorFlow, NLP, Keras, LSTM, NLTK
Dataset: IMDB Movie Reviews (50,000 reviews)

Task: Binary Sentiment Classification
      - Positive Review (1)
      - Negative Review (0)
============================================================
"""

# =============================================
# 1. IMPORT ALL LIBRARIES
# =============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
print("📥 Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"TensorFlow Version : {tf.__version__}")
print(f"NLTK Version       : {nltk.__version__}")
print(f"GPU Available      : {tf.config.list_physical_devices('GPU')}")
print("=" * 60)


# =============================================
# 2. LOAD DATASET
# =============================================
print("\n📦 Loading IMDB Dataset...")

# IMDB Dataset from Keras - 50,000 movie reviews
# Already encoded as sequences of word indices
VOCAB_SIZE = 20000  # Top 20,000 most frequent words

(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)

print(f"\n{'='*60}")
print("DATASET INFORMATION")
print(f"{'='*60}")
print(f"Training Samples  : {len(X_train_raw)}")
print(f"Testing Samples   : {len(X_test_raw)}")
print(f"Vocabulary Size   : {VOCAB_SIZE}")
print(f"Classes           : 2 (Positive, Negative)")
print(f"Training Positive : {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Training Negative : {len(y_train) - sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)")
print(f"{'='*60}")

# Get word index mapping
word_index = keras.datasets.imdb.get_word_index()

# Create reverse mapping (index -> word)
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = '<UNUSED>'


def decode_review(encoded_review):
    """Convert encoded review back to text"""
    return ' '.join(
        reverse_word_index.get(idx, '?') for idx in encoded_review
    )


# Show sample reviews
print("\n📝 Sample Reviews:")
for i in range(3):
    decoded = decode_review(X_train_raw[i])
    sentiment = "Positive ✅" if y_train[i] == 1 else "Negative ❌"
    print(f"\n--- Review {i+1} ({sentiment}) ---")
    print(f"Encoded (first 20 words): {X_train_raw[i][:20]}...")
    print(f"Decoded (first 100 chars): {decoded[:100]}...")
    print(f"Review Length: {len(X_train_raw[i])} words")


# =============================================
# 3. DATA EXPLORATION & VISUALIZATION
# =============================================
print("\n📊 Exploring Dataset...")


def plot_data_exploration(X_train, y_train, X_test, y_test):
    """Comprehensive data exploration visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IMDB Dataset Exploration',
                 fontsize=18, fontweight='bold')

    # 1. Class Distribution - Training
    labels = ['Negative', 'Positive']
    train_counts = [
        len(y_train) - sum(y_train),
        sum(y_train)
    ]
    colors = ['#ff6b6b', '#51cf66']

    axes[0][0].bar(labels, train_counts, color=colors,
                   edgecolor='black', width=0.5)
    axes[0][0].set_title('Training Set - Class Distribution',
                         fontsize=13, fontweight='bold')
    axes[0][0].set_ylabel('Count')
    for i, count in enumerate(train_counts):
        axes[0][0].text(i, count + 200, str(count),
                        ha='center', fontsize=12, fontweight='bold')

    # 2. Class Distribution - Testing
    test_counts = [
        len(y_test) - sum(y_test),
        sum(y_test)
    ]
    axes[0][1].bar(labels, test_counts, color=colors,
                   edgecolor='black', width=0.5)
    axes[0][1].set_title('Testing Set - Class Distribution',
                         fontsize=13, fontweight='bold')
    axes[0][1].set_ylabel('Count')
    for i, count in enumerate(test_counts):
        axes[0][1].text(i, count + 200, str(count),
                        ha='center', fontsize=12, fontweight='bold')

    # 3. Review Length Distribution
    train_lengths = [len(review) for review in X_train]
    axes[1][0].hist(train_lengths, bins=50, color='steelblue',
                    edgecolor='black', alpha=0.7)
    axes[1][0].axvline(np.mean(train_lengths), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(train_lengths):.0f}')
    axes[1][0].axvline(np.median(train_lengths), color='green',
                       linestyle='--', linewidth=2,
                       label=f'Median: {np.median(train_lengths):.0f}')
    axes[1][0].set_title('Review Length Distribution',
                         fontsize=13, fontweight='bold')
    axes[1][0].set_xlabel('Number of Words')
    axes[1][0].set_ylabel('Frequency')
    axes[1][0].legend()

    # 4. Review Length by Sentiment
    pos_lengths = [
        len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 1
    ]
    neg_lengths = [
        len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 0
    ]
    axes[1][1].boxplot(
        [neg_lengths, pos_lengths],
        labels=['Negative', 'Positive'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue')
    )
    axes[1][1].set_title('Review Length by Sentiment',
                         fontsize=13, fontweight='bold')
    axes[1][1].set_ylabel('Number of Words')

    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Data exploration saved as 'data_exploration.png'")

    # Print statistics
    print(f"\n📊 Review Length Statistics:")
    print(f"   Mean Length   : {np.mean(train_lengths):.0f} words")
    print(f"   Median Length : {np.median(train_lengths):.0f} words")
    print(f"   Min Length    : {np.min(train_lengths)} words")
    print(f"   Max Length    : {np.max(train_lengths)} words")
    print(f"   Std Deviation : {np.std(train_lengths):.0f} words")


plot_data_exploration(X_train_raw, y_train, X_test_raw, y_test)


# =============================================
# 4. TEXT PREPROCESSING WITH NLP
# =============================================
print("\n" + "=" * 60)
print("⚙️  TEXT PREPROCESSING WITH NLP")
print("=" * 60)

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print(f"\n📋 NLP Tools Initialized:")
print(f"   Stemmer      : PorterStemmer")
print(f"   Lemmatizer   : WordNetLemmatizer")
print(f"   Stop Words   : {len(stop_words)} words")
print(f"   Sample Stop Words: {list(stop_words)[:10]}")


def preprocess_text(text):
    """
    Complete NLP text preprocessing pipeline:
    1. Lowercase conversion
    2. Remove HTML tags
    3. Remove URLs
    4. Remove special characters & numbers
    5. Tokenization
    6. Remove stopwords
    7. Stemming
    8. Lemmatization
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 4. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Tokenization
    tokens = word_tokenize(text)

    # 7. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # 9. Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return {
        'original_tokens': tokens,
        'stemmed': stemmed_tokens,
        'lemmatized': lemmatized_tokens,
        'processed_text': ' '.join(lemmatized_tokens)
    }


# Demonstrate preprocessing on sample review
sample_review = decode_review(X_train_raw[0])
processed = preprocess_text(sample_review)

print(f"\n{'='*60}")
print("TEXT PREPROCESSING DEMONSTRATION")
print(f"{'='*60}")
print(f"\n📝 Original Text (first 200 chars):")
print(f"   {sample_review[:200]}...")
print(f"\n🔤 After Tokenization (first 15 tokens):")
print(f"   {processed['original_tokens'][:15]}")
print(f"\n🌿 After Stemming (first 15 tokens):")
print(f"   {processed['stemmed'][:15]}")
print(f"\n📖 After Lemmatization (first 15 tokens):")
print(f"   {processed['lemmatized'][:15]}")
print(f"\n✅ Final Processed Text (first 200 chars):")
print(f"   {processed['processed_text'][:200]}...")


# Decode all reviews and preprocess
print("\n🔄 Decoding and preprocessing all reviews...")
print("   (This may take a few minutes...)")

# Decode reviews from indices to text
train_texts = [decode_review(review) for review in X_train_raw]
test_texts = [decode_review(review) for review in X_test_raw]

# Preprocess all texts
train_processed = []
for i, text in enumerate(train_texts):
    result = preprocess_text(text)
    train_processed.append(result['processed_text'])
    if (i + 1) % 5000 == 0:
        print(f"   Processed {i+1}/{len(train_texts)} training reviews...")

test_processed = []
for i, text in enumerate(test_texts):
    result = preprocess_text(text)
    test_processed.append(result['processed_text'])
    if (i + 1) % 5000 == 0:
        print(f"   Processed {i+1}/{len(test_texts)} testing reviews...")

print("✅ All reviews preprocessed!")


# =============================================
# 5. WORD CLOUD VISUALIZATION
# =============================================
print("\n☁️  Generating Word Clouds...")

try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Word Clouds by Sentiment',
                 fontsize=18, fontweight='bold')

    # Positive reviews word cloud
    positive_text = ' '.join(
        [train_processed[i] for i in range(len(train_processed))
         if y_train[i] == 1]
    )
    wc_pos = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Greens',
        max_words=100
    ).generate(positive_text)

    axes[0].imshow(wc_pos, interpolation='bilinear')
    axes[0].set_title('Positive Reviews', fontsize=14,
                      fontweight='bold', color='green')
    axes[0].axis('off')

    # Negative reviews word cloud
    negative_text = ' '.join(
        [train_processed[i] for i in range(len(train_processed))
         if y_train[i] == 0]
    )
    wc_neg = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(negative_text)

    axes[1].imshow(wc_neg, interpolation='bilinear')
    axes[1].set_title('Negative Reviews', fontsize=14,
                      fontweight='bold', color='red')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('word_clouds.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Word clouds saved as 'word_clouds.png'")

except ImportError:
    print("⚠️  WordCloud not installed. Skipping visualization.")
    print("   Install with: pip install wordcloud")


# =============================================
# 6. TOKENIZATION & SEQUENCE PADDING
# =============================================
print("\n" + "=" * 60)
print("🔢 TOKENIZATION & SEQUENCE PADDING")
print("=" * 60)

MAX_WORDS = 20000    # Maximum vocabulary size
MAX_LEN = 200        # Maximum sequence length (pad/truncate to this)
EMBEDDING_DIM = 128  # Word embedding dimension

# Create and fit tokenizer
print(f"\n📋 Tokenizer Configuration:")
print(f"   Max Vocabulary : {MAX_WORDS}")
print(f"   Max Sequence   : {MAX_LEN}")
print(f"   Embedding Dim  : {EMBEDDING_DIM}")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token='<OOV>'  # Out of vocabulary token
)

# Fit tokenizer on training data
tokenizer.fit_on_texts(train_processed)

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(train_processed)
X_test_sequences = tokenizer.texts_to_sequences(test_processed)

print(f"\n📊 Tokenizer Statistics:")
print(f"   Total Unique Words : {len(tokenizer.word_index)}")
print(f"   Words Used (top)   : {MAX_WORDS}")

# Show tokenization example
print(f"\n📝 Tokenization Example:")
print(f"   Text : '{train_processed[0][:80]}...'")
print(f"   Sequence (first 20): {X_train_sequences[0][:20]}")

# Pad sequences - make all same length
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LEN,
    padding='post',      # Pad at the end
    truncating='post'    # Truncate from end if too long
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

print(f"\n📐 After Padding:")
print(f"   Training Shape : {X_train_padded.shape}")
print(f"   Testing Shape  : {X_test_padded.shape}")
print(f"   Sequence Length : {MAX_LEN} (fixed)")

# Show padding example
print(f"\n📝 Padding Example (last 20 values):")
print(f"   Before padding length: {len(X_train_sequences[0])}")
print(f"   After padding (tail) : {X_train_padded[0][-20:]}")
print(f"   (0s = padding)")

# Validation split
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_padded, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"\n📊 Final Data Splits:")
print(f"   Training   : {X_train_final.shape[0]} samples")
print(f"   Validation : {X_val_final.shape[0]} samples")
print(f"   Testing    : {X_test_padded.shape[0]} samples")

# Save tokenizer
os.makedirs('saved_models', exist_ok=True)
with open('saved_models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("\n✅ Tokenizer saved as 'saved_models/tokenizer.pickle'")


# =============================================
# 7. VISUALIZE SEQUENCE LENGTH DISTRIBUTION
# =============================================
def plot_sequence_analysis(sequences, max_len):
    """Visualize sequence lengths and padding"""
    lengths = [len(seq) for seq in sequences]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Sequence Analysis', fontsize=16, fontweight='bold')

    # Length distribution
    axes[0].hist(lengths, bins=50, color='steelblue',
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(max_len, color='red', linestyle='--',
                    linewidth=2, label=f'Max Length = {max_len}')
    axes[0].axvline(np.mean(lengths), color='green', linestyle='--',
                    linewidth=2, label=f'Mean = {np.mean(lengths):.0f}')
    axes[0].set_title('Sequence Length Distribution', fontsize=13)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Coverage analysis
    coverage = [
        sum(1 for l in lengths if l <= threshold) / len(lengths) * 100
        for threshold in range(50, 501, 10)
    ]
    thresholds = list(range(50, 501, 10))
    axes[1].plot(thresholds, coverage, linewidth=2, color='steelblue')
    axes[1].axvline(max_len, color='red', linestyle='--',
                    linewidth=2, label=f'Chosen = {max_len}')
    axes[1].axhline(
        sum(1 for l in lengths if l <= max_len) / len(lengths) * 100,
        color='green', linestyle=':', linewidth=1
    )
    axes[1].set_title('Coverage vs Max Length', fontsize=13)
    axes[1].set_xlabel('Max Sequence Length')
    axes[1].set_ylabel('Coverage (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sequence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    coverage_at_max = sum(
        1 for l in lengths if l <= max_len
    ) / len(lengths) * 100
    print(f"✅ At MAX_LEN={max_len}: {coverage_at_max:.1f}% reviews fully covered")


plot_sequence_analysis(X_train_sequences, MAX_LEN)


# =============================================
# 8. BUILD SIMPLE RNN MODEL
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 1: SIMPLE RNN")
print("=" * 60)


def build_simple_rnn():
    """
    Simple RNN Architecture:
    - Embedding Layer (word vectors)
    - SimpleRNN Layer
    - Dense output
    """
    model = models.Sequential(name="Simple_RNN_Model")

    # Embedding Layer - converts word indices to dense vectors
    model.add(layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        name='embedding'
    ))

    # Simple RNN Layer
    model.add(layers.SimpleRNN(
        64,
        dropout=0.3,
        recurrent_dropout=0.3,
        return_sequences=False,
        name='simple_rnn'
    ))

    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    return model


rnn_model = build_simple_rnn()

rnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Simple RNN Architecture:")
rnn_model.summary()


# =============================================
# 9. BUILD LSTM MODEL
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 2: LSTM (Long Short-Term Memory)")
print("=" * 60)


def build_lstm_model():
    """
    LSTM Architecture:
    - Embedding Layer
    - LSTM Layer (handles long-term dependencies)
    - Dense output with sigmoid
    """
    model = models.Sequential(name="LSTM_Model")

    # Embedding Layer
    model.add(layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        name='embedding'
    ))

    # LSTM Layer
    model.add(layers.LSTM(
        128,
        dropout=0.3,
        recurrent_dropout=0.3,
        return_sequences=False,
        name='lstm'
    ))

    # Dense Layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.BatchNormalization(name='batch_norm1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(32, activation='relu', name='dense2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    return model


lstm_model = build_lstm_model()

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 LSTM Architecture:")
lstm_model.summary()


# =============================================
# 10. BUILD BIDIRECTIONAL LSTM MODEL
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 3: BIDIRECTIONAL LSTM")
print("=" * 60)


def build_bidirectional_lstm():
    """
    Bidirectional LSTM Architecture:
    - Reads sequence forward AND backward
    - Captures context from both directions
    - Stacked Bi-LSTM layers
    """
    model = models.Sequential(name="Bidirectional_LSTM_Model")

    # Embedding Layer
    model.add(layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        name='embedding'
    ))

    # Bidirectional LSTM - Layer 1 (return sequences for stacking)
    model.add(layers.Bidirectional(
        layers.LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=True  # Output sequence for next LSTM
        ),
        name='bi_lstm_1'
    ))

    # Bidirectional LSTM - Layer 2
    model.add(layers.Bidirectional(
        layers.LSTM(
            32,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=False
        ),
        name='bi_lstm_2'
    ))

    # Dense Layers
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.BatchNormalization(name='batch_norm1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(32, activation='relu', name='dense2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    return model


bilstm_model = build_bidirectional_lstm()

bilstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Bidirectional LSTM Architecture:")
bilstm_model.summary()


# =============================================
# 11. TRAINING CALLBACKS
# =============================================
print("\n⚙️  Setting up Training Callbacks...")


def get_callbacks(model_name):
    """Create callbacks for each model"""
    return [
        callbacks.ModelCheckpoint(
            f'saved_models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]


BATCH_SIZE = 128
EPOCHS = 20  # EarlyStopping will manage actual epochs


# =============================================
# 12. TRAIN ALL MODELS
# =============================================

# ---------- Train Simple RNN ----------
print("\n" + "=" * 60)
print("🚂 TRAINING MODEL 1: Simple RNN")
print("=" * 60)

rnn_history = rnn_model.fit(
    X_train_final, y_train_final,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_final, y_val_final),
    callbacks=get_callbacks('simple_rnn'),
    verbose=1
)

print("\n✅ Simple RNN Training Complete!")

# ---------- Train LSTM ----------
print("\n" + "=" * 60)
print("🚂 TRAINING MODEL 2: LSTM")
print("=" * 60)

lstm_history = lstm_model.fit(
    X_train_final, y_train_final,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_final, y_val_final),
    callbacks=get_callbacks('lstm'),
    verbose=1
)

print("\n✅ LSTM Training Complete!")

# ---------- Train Bidirectional LSTM ----------
print("\n" + "=" * 60)
print("🚂 TRAINING MODEL 3: Bidirectional LSTM")
print("=" * 60)

bilstm_history = bilstm_model.fit(
    X_train_final, y_train_final,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_final, y_val_final),
    callbacks=get_callbacks('bidirectional_lstm'),
    verbose=1
)

print("\n✅ Bidirectional LSTM Training Complete!")


# =============================================
# 13. TRAINING HISTORY VISUALIZATION
# =============================================
def plot_training_history(histories, model_names):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Training History - All Models',
                 fontsize=18, fontweight='bold')

    colors = ['blue', 'green', 'purple']

    for idx, (history, name, color) in enumerate(
        zip(histories, model_names, colors)
    ):
        # Accuracy
        axes[0][idx].plot(history.history['accuracy'],
                         label='Train', linewidth=2, color=color)
        axes[0][idx].plot(history.history['val_accuracy'],
                         label='Validation', linewidth=2,
                         color='red', linestyle='--')
        axes[0][idx].set_title(f'{name}\nAccuracy',
                              fontsize=12, fontweight='bold')
        axes[0][idx].set_xlabel('Epoch')
        axes[0][idx].set_ylabel('Accuracy')
        axes[0][idx].legend()
        axes[0][idx].grid(True, alpha=0.3)
        axes[0][idx].set_ylim([0.5, 1.0])

        # Loss
        axes[1][idx].plot(history.history['loss'],
                         label='Train', linewidth=2, color=color)
        axes[1][idx].plot(history.history['val_loss'],
                         label='Validation', linewidth=2,
                         color='red', linestyle='--')
        axes[1][idx].set_title(f'{name}\nLoss',
                              fontsize=12, fontweight='bold')
        axes[1][idx].set_xlabel('Epoch')
        axes[1][idx].set_ylabel('Loss')
        axes[1][idx].legend()
        axes[1][idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('all_models_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Training history saved as 'all_models_training_history.png'")


plot_training_history(
    [rnn_history, lstm_history, bilstm_history],
    ['Simple RNN', 'LSTM', 'Bidirectional LSTM']
)


# =============================================
# 14. MODEL EVALUATION
# =============================================
print("\n" + "=" * 60)
print("📊 MODEL EVALUATION ON TEST SET")
print("=" * 60)


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

    print(f"\n{'='*50}")
    print(f"📊 {model_name} - Test Results")
    print(f"{'='*50}")
    print(f"   Accuracy  : {accuracy * 100:.2f}%")
    print(f"   Precision : {precision * 100:.2f}%")
    print(f"   Recall    : {recall * 100:.2f}%")
    print(f"   F1-Score  : {f1 * 100:.2f}%")
    print(f"   Loss      : {test_loss:.4f}")

    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive'],
        digits=4
    ))

    return {
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': test_loss,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }


# Evaluate all models
rnn_results = evaluate_model(
    rnn_model, X_test_padded, y_test, "Simple RNN"
)
lstm_results = evaluate_model(
    lstm_model, X_test_padded, y_test, "LSTM"
)
bilstm_results = evaluate_model(
    bilstm_model, X_test_padded, y_test, "Bidirectional LSTM"
)

all_results = [rnn_results, lstm_results, bilstm_results]


# =============================================
# 15. CONFUSION MATRIX VISUALIZATION
# =============================================
print("\n📊 Generating Confusion Matrices...")


def plot_all_confusion_matrices(results_list, y_test):
    """Plot confusion matrices for all models side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices - All Models',
                 fontsize=18, fontweight='bold')

    for idx, results in enumerate(results_list):
        cm = confusion_matrix(y_test, results['y_pred'])

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[idx],
            linewidths=0.5,
            square=True,
            annot_kws={"size": 14}
        )
        axes[idx].set_title(
            f"{results['name']}\nAccuracy: {results['accuracy']*100:.2f}%",
            fontsize=12, fontweight='bold'
        )
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrices saved as 'confusion_matrices.png'")


plot_all_confusion_matrices(all_results, y_test)


# =============================================
# 16. ROC CURVE & AUC
# =============================================
print("\n📈 Generating ROC Curves...")


def plot_roc_curves(results_list, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'purple']

    for results, color in zip(results_list, colors):
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_prob'])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, color=color, linewidth=2,
            label=f"{results['name']} (AUC = {roc_auc:.4f})"
        )

    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linewidth=1,
             linestyle='--', label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves - Model Comparison',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ ROC curves saved as 'roc_curves.png'")


plot_roc_curves(all_results, y_test)


# =============================================
# 17. MODEL COMPARISON VISUALIZATION
# =============================================
print("\n📊 Model Comparison...")


def plot_model_comparison(results_list):
    """Comprehensive model comparison bar chart"""
    model_names = [r['name'] for r in results_list]
    metrics = {
        'Accuracy': [r['accuracy'] * 100 for r in results_list],
        'Precision': [r['precision'] * 100 for r in results_list],
        'Recall': [r['recall'] * 100 for r in results_list],
        'F1-Score': [r['f1'] * 100 for r in results_list]
    }

    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width,
                      label=metric_name, color=colors[i],
                      edgecolor='black')
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_xlabel('Models', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Model Comparison - All Metrics',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([70, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Model comparison saved as 'model_comparison.png'")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}")
    print(f"{'='*70}")
    for r in results_list:
        print(f"{r['name']:<25} {r['accuracy']*100:>9.2f}% "
              f"{r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% "
              f"{r['f1']*100:>9.2f}%")
    print(f"{'='*70}")


plot_model_comparison(all_results)


# =============================================
# 18. PREDICTION ON CUSTOM TEXT
# =============================================
print("\n" + "=" * 60)
print("🔮 PREDICTION ON CUSTOM TEXT")
print("=" * 60)


def predict_sentiment(text, model, tokenizer, model_name="Model",
                      max_len=MAX_LEN):
    """
    Predict sentiment for any custom text input.

    Pipeline:
    1. Preprocess text (NLP pipeline)
    2. Tokenize
    3. Pad sequence
    4. Predict
    """
    # Step 1: Preprocess
    processed = preprocess_text(text)
    processed_text = processed['processed_text']

    # Step 2: Tokenize
    sequence = tokenizer.texts_to_sequences([processed_text])

    # Step 3: Pad
    padded = pad_sequences(sequence, maxlen=max_len,
                           padding='post', truncating='post')

    # Step 4: Predict
    prediction = model.predict(padded, verbose=0)[0][0]

    # Result
    sentiment = "POSITIVE ✅" if prediction > 0.5 else "NEGATIVE ❌"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"\n{'─'*50}")
    print(f"📝 Input Text  : {text[:80]}...")
    print(f"⚙️  Processed   : {processed_text[:80]}...")
    print(f"🤖 Model       : {model_name}")
    print(f"📊 Raw Score   : {prediction:.4f}")
    print(f"🎯 Sentiment   : {sentiment}")
    print(f"💪 Confidence  : {confidence * 100:.2f}%")
    print(f"{'─'*50}")

    return sentiment, confidence


# Test with custom reviews
custom_reviews = [
    "This movie was absolutely amazing! The acting was superb and the story was captivating. I loved every minute of it!",
    "Terrible movie. Waste of time and money. The plot made no sense and acting was awful. Would not recommend to anyone.",
    "The film was okay, nothing special. Some parts were good but overall it was average and forgettable.",
    "One of the best movies I have ever seen! Brilliant direction, outstanding performances, and a touching story.",
    "I fell asleep halfway through. Boring, predictable, and poorly made. The worst movie of the year without doubt.",
    "A masterpiece of cinema! Every scene was beautifully crafted. The soundtrack was phenomenal and emotionally powerful."
]

print("\n🔮 Testing with Custom Reviews:")
print("=" * 60)

# Use the best model (Bidirectional LSTM) for predictions
best_model = bilstm_model
best_model_name = "Bidirectional LSTM"

for review in custom_reviews:
    predict_sentiment(review, best_model, tokenizer, best_model_name)


# =============================================
# 19. INTERACTIVE PREDICTION (Bonus)
# =============================================
def interactive_prediction(model, tokenizer, model_name):
    """Interactive mode for real-time predictions"""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE SENTIMENT PREDICTION")
    print("   Type a movie review and get instant prediction!")
    print("   Type 'quit' to exit")
    print("=" * 60)

    while True:
        user_input = input("\n📝 Enter review: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if len(user_input) < 5:
            print("⚠️  Please enter a longer review!")
            continue

        predict_sentiment(user_input, model, tokenizer, model_name)


# Uncomment below line to enable interactive mode
# interactive_prediction(bilstm_model, tokenizer, "Bidirectional LSTM")


# =============================================
# 20. EMBEDDING VISUALIZATION
# =============================================
print("\n🔍 Visualizing Word Embeddings...")


def visualize_embeddings(model, tokenizer, num_words=100):
    """Visualize word embeddings using PCA"""
    from sklearn.decomposition import PCA

    # Get embedding layer weights
    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()[0]

    print(f"Embedding Matrix Shape: {embedding_weights.shape}")
    print(f"(Vocabulary Size × Embedding Dimension)")

    # Get most common words
    word_freq = sorted(
        tokenizer.word_index.items(),
        key=lambda x: x[1]
    )[:num_words]

    words = [w[0] for w in word_freq]
    word_indices = [w[1] for w in word_freq]
    word_vectors = embedding_weights[word_indices]

    # PCA to reduce to 2D
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)

    plt.figure(figsize=(16, 12))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1],
                c='steelblue', alpha=0.6, s=50)

    # Annotate words
    for i, word in enumerate(words[:50]):  # Label top 50
        plt.annotate(
            word,
            xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
            fontsize=9, alpha=0.8
        )

    plt.title('Word Embedding Visualization (PCA)',
              fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word_embeddings_pca.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Word embeddings visualization saved!")


visualize_embeddings(bilstm_model, tokenizer)


# =============================================
# 21. SAVE ALL MODELS
# =============================================
print("\n" + "=" * 60)
print("💾 SAVING ALL MODELS")
print("=" * 60)

# Save all models
rnn_model.save('saved_models/simple_rnn_model.h5')
print("✅ Simple RNN saved: saved_models/simple_rnn_model.h5")

lstm_model.save('saved_models/lstm_model.h5')
print("✅ LSTM saved: saved_models/lstm_model.h5")

bilstm_model.save('saved_models/bidirectional_lstm_model.h5')
print("✅ Bi-LSTM saved: saved_models/bidirectional_lstm_model.h5")

print(f"\n📖 Load models like this:")
print(f"   model = keras.models.load_model('saved_models/lstm_model.h5')")
print(f"   tokenizer = pickle.load(open('saved_models/tokenizer.pickle', 'rb'))")


# =============================================
# 22. FINAL SUMMARY
# =============================================
# Find best model
best = max(all_results, key=lambda x: x['accuracy'])

print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE - FINAL SUMMARY")
print("=" * 60)
print(f"""
╔══════════════════════════════════════════════════════════════╗
║     SENTIMENT ANALYSIS USING RNN & NLP                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Dataset        : IMDB Movie Reviews (50,000 reviews)       ║
║  Task           : Binary Sentiment Classification           ║
║  Vocabulary     : {MAX_WORDS:,} words                            ║
║  Sequence Length : {MAX_LEN} (padded)                            ║
║  Embedding Dim  : {EMBEDDING_DIM}                                    ║
║                                                              ║
║  NLP Preprocessing Pipeline:                                 ║
║  ✅ Lowercase conversion                                     ║
║  ✅ HTML tag removal                                         ║
║  ✅ Special character removal                                ║
║  ✅ Tokenization (NLTK word_tokenize)                        ║
║  ✅ Stopword removal ({len(stop_words)} stop words)                     ║
║  ✅ Stemming (PorterStemmer)                                 ║
║  ✅ Lemmatization (WordNetLemmatizer)                        ║
║  ✅ Word Embeddings (Learned)                                ║
║  ✅ Sequence Padding                                         ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ MODEL 1: Simple RNN                                  │    ║
║  │ Accuracy : {rnn_results['accuracy']*100:>6.2f}%    F1-Score : {rnn_results['f1']*100:>6.2f}%       │    ║
║  │ Precision: {rnn_results['precision']*100:>6.2f}%    Recall   : {rnn_results['recall']*100:>6.2f}%       │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ MODEL 2: LSTM                                        │    ║
║  │ Accuracy : {lstm_results['accuracy']*100:>6.2f}%    F1-Score : {lstm_results['f1']*100:>6.2f}%       │    ║
║  │ Precision: {lstm_results['precision']*100:>6.2f}%    Recall   : {lstm_results['recall']*100:>6.2f}%       │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ MODEL 3: Bidirectional LSTM  ⭐ BEST                 │    ║
║  │ Accuracy : {bilstm_results['accuracy']*100:>6.2f}%    F1-Score : {bilstm_results['f1']*100:>6.2f}%       │    ║
║  │ Precision: {bilstm_results['precision']*100:>6.2f}%    Recall   : {bilstm_results['recall']*100:>6.2f}%       │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  🏆 Best Model: {best['name']:<40} ║
║     Best Accuracy: {best['accuracy']*100:.2f}%                              ║
║                                                              ║
║  Files Generated:                                            ║
║  📄 data_exploration.png                                     ║
║  📄 word_clouds.png                                          ║
║  📄 sequence_analysis.png                                    ║
║  📄 all_models_training_history.png                          ║
║  📄 confusion_matrices.png                                   ║
║  📄 roc_curves.png                                           ║
║  📄 model_comparison.png                                     ║
║  📄 word_embeddings_pca.png                                  ║
║  📦 saved_models/simple_rnn_model.h5                         ║
║  📦 saved_models/lstm_model.h5                               ║
║  📦 saved_models/bidirectional_lstm_model.h5                 ║
║  📦 saved_models/tokenizer.pickle                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("🎯 Sentiment Analysis Project Complete!")
print("   All 3 models trained, evaluated, and saved!")
print("   Ready for interview submission! 💪")