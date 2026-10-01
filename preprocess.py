"""
preprocess.py
-------------
Step 2: Preprocess MIDI data into note sequences suitable for training,
using music21 (as suggested in the task) to parse each MIDI file into a
stream of notes and chords, then encode them as integers for an RNN.

Usage:
    python preprocess.py --midi_dir data/midi --seq_len 50
"""
import argparse
import glob
import os
import pickle

import numpy as np
from music21 import converter, instrument, note, chord


def extract_notes(midi_dir: str):
    """Parse every .mid file into a flat list of note/chord name strings."""
    all_notes = []
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid"))
    if not midi_files:
        raise FileNotFoundError(
            f"No .mid files found in '{midi_dir}'. Run collect_data.py first."
        )

    print(f"Parsing {len(midi_files)} MIDI files with music21...")
    for i, file in enumerate(midi_files):
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"  could not parse {file}: {e}")
            continue

        # Flatten the whole score into one chronological stream of notes/chords.
        # (Using every part rather than just parts[0] avoids missing notes on
        # scores where the first "part" is just instrument/clef metadata.)
        elements = midi.flatten().notes

        for el in elements:
            if isinstance(el, note.Note):
                # e.g. "C4"
                all_notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                # encode a chord as its pitches joined by dots, e.g. "4.7.11"
                all_notes.append(".".join(str(n) for n in el.normalOrder))
        if (i + 1) % 10 == 0:
            print(f"  parsed {i + 1}/{len(midi_files)} files, {len(all_notes)} events so far")

    print(f"Extracted {len(all_notes)} total note/chord events.")
    return all_notes


def build_sequences(all_notes, seq_len: int):
    """Turn the flat note list into (input_sequence -> next_note) training pairs."""
    vocab = sorted(set(all_notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}

    network_input = []
    network_output = []
    for i in range(len(all_notes) - seq_len):
        seq_in = all_notes[i:i + seq_len]
        seq_out = all_notes[i + seq_len]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_vocab = len(vocab)
    X = np.reshape(network_input, (len(network_input), seq_len, 1))
    X = X / float(n_vocab)  # normalize for the LSTM
    y = np.eye(n_vocab)[network_output]  # one-hot encode targets

    return X, y, note_to_int, n_vocab


def main(midi_dir: str, seq_len: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_notes = extract_notes(midi_dir)
    X, y, note_to_int, n_vocab = build_sequences(all_notes, seq_len)

    print(f"Vocabulary size: {n_vocab} unique notes/chords")
    print(f"Training sequences: X={X.shape}, y={y.shape}")

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({
            "note_to_int": note_to_int,
            "int_to_note": {i: n for n, i in note_to_int.items()},
            "n_vocab": n_vocab,
            "seq_len": seq_len,
        }, f)

    print(f"Saved preprocessed data to '{out_dir}/' (X.npy, y.npy, mappings.pkl)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, default="data/midi")
    parser.add_argument("--seq_len", type=int, default=50, help="Notes per input sequence")
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()
    main(args.midi_dir, args.seq_len, args.out)
