"""
train_model.py
---------------
Step 3: Build and train a deep learning model (stacked LSTM) to learn
patterns in the note sequences produced by preprocess.py.

Requires TensorFlow (pip install tensorflow). Training is CPU-feasible for
a small dataset/epoch count, but a GPU will make it much faster.

Usage:
    python train_model.py --epochs 50 --batch_size 64
"""
import argparse
import os
import pickle

import numpy as np


def build_model(seq_len: int, n_vocab: int):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(n_vocab, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def main(data_dir: str, model_dir: str, epochs: int, batch_size: int):
    from tensorflow.keras.callbacks import ModelCheckpoint

    os.makedirs(model_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

    seq_len = mappings["seq_len"]
    n_vocab = mappings["n_vocab"]
    print(f"Loaded {X.shape[0]} training sequences | seq_len={seq_len} | vocab={n_vocab}")

    model = build_model(seq_len, n_vocab)

    checkpoint_path = os.path.join(model_dir, "music_lstm_best.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, verbose=1)

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    final_path = os.path.join(model_dir, "music_lstm_final.keras")
    model.save(final_path)
    print(f"Training complete. Best model: {checkpoint_path} | Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.epochs, args.batch_size)
