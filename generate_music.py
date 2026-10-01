"""
generate_music.py
------------------
Step 5: Use the trained model to generate a brand-new note sequence, then
convert it back into real notes/chords with music21 and save/play it as a
MIDI file.

Usage:
    python generate_music.py --n_notes 200 --out output/generated.mid
"""
import argparse
import os
import pickle

import numpy as np


def generate_notes(model, network_input, int_to_note, n_vocab, n_notes, temperature=1.0):
    """Autoregressively sample n_notes new events from the trained model."""
    start = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start].flatten())  # seed sequence, already normalized to [0,1)
    seq_len = len(pattern)

    prediction_output = []
    for _ in range(n_notes):
        input_seq = np.reshape(pattern, (1, seq_len, 1))
        prediction = model.predict(input_seq, verbose=0)[0]

        # temperature-scaled sampling for more/less "creative" output
        prediction = np.log(prediction + 1e-9) / temperature
        probs = np.exp(prediction) / np.sum(np.exp(prediction))
        index = np.random.choice(len(probs), p=probs)

        prediction_output.append(int_to_note[index])

        pattern.append(index / float(n_vocab))
        pattern = pattern[1:]

    return prediction_output


def sequence_to_midi(prediction_output, out_path: str):
    from music21 import stream, note, chord, instrument

    output_stream = stream.Part()
    output_stream.insert(0, instrument.Piano())

    offset = 0.0
    for pattern in prediction_output:
        if "." in pattern:  # a chord, e.g. "4.7.11"
            notes_in_chord = [note.Note(int(n)) for n in pattern.split(".")]
            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:  # a single note, e.g. "C4"
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_stream.append(new_note)
        offset += 0.5  # eighth-note spacing between generated events

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    output_stream.write("midi", fp=out_path)
    print(f"Saved generated music to '{out_path}'")


def main(model_path: str, data_dir: str, n_notes: int, temperature: float, out_path: str):
    from tensorflow.keras.models import load_model

    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    X = np.load(os.path.join(data_dir, "X.npy"))

    model = load_model(model_path)
    prediction_output = generate_notes(
        model, X, mappings["int_to_note"], mappings["n_vocab"], n_notes, temperature
    )
    sequence_to_midi(prediction_output, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/music_lstm_best.keras")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n_notes", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Lower = safer/more repetitive, higher = more random")
    parser.add_argument("--out", type=str, default="output/generated.mid")
    args = parser.parse_args()
    main(args.model, args.data_dir, args.n_notes, args.temperature, args.out)
