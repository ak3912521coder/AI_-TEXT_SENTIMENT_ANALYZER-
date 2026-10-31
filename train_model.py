import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)


max_features = 10000
maxlen = 100  

print("Loading IMDB dataset...")
(X_train_text, y_train), (X_test_text, y_test) = imdb.load_data(num_words=max_features)

X_train = pad_sequences(X_train_text, maxlen=maxlen, padding="post", truncating="post")
X_test = pad_sequences(X_test_text, maxlen=maxlen, padding="post", truncating="post")

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


model = Sequential([
    Embedding(
        input_dim=max_features,
        output_dim=64,  # Embedding dimension badha diya hai
        input_length=maxlen
    ),
    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), # Stronger LSTM layer
    LSTM(32, dropout=0.3),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid") # Binary classification (0 = Negative, 1 = Positive)
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


print("\nTraining model... (Yeh kuch minutes lega)")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

predictions = (
    model.predict(X_test, verbose=0).ravel() >= 0.5
).astype(int)

print("\n----------------------------------------")
print(f"Final Test Accuracy: {round(test_accuracy * 100, 2)}%")
print("----------------------------------------")


model.save("sentiment_model_imdb.keras")
print("\nModel saved successfully as 'sentiment_model_imdb.keras'")


plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM IMDB Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_imdb.png")
plt.show()