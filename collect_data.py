"""
collect_data.py
----------------
Step 1: Collect MIDI music data to train the model.

music21 ships with a bundled corpus of public-domain classical pieces
(433 Bach chorales) so this script needs NO internet download and no
external dataset — it just exports a subset of that corpus to real .mid
files in data/midi/, which the rest of the pipeline treats like any other
MIDI dataset you might collect yourself (e.g. from https://www.kaggle.com/datasets
or https://magenta.tensorflow.org/datasets/ for jazz/pop MIDI collections).

Usage:
    python collect_data.py --n 40
"""
import argparse
import os
from music21 import corpus

def main(n_pieces: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bach_paths = corpus.getComposer('bach')[:n_pieces]

    print(f"Exporting {len(bach_paths)} Bach chorales to '{out_dir}/' as MIDI...")
    exported = 0
    for path in bach_paths:
        try:
            score = corpus.parse(path)
            name = os.path.splitext(os.path.basename(str(path)))[0]
            out_path = os.path.join(out_dir, f"{name}.mid")
            score.write('midi', fp=out_path)
            exported += 1
        except Exception as e:
            print(f"  skipped {path}: {e}")

    print(f"Done. {exported} MIDI files saved to '{out_dir}/'.")
    print("Swap in your own dataset by dropping more .mid files into this folder —")
    print("preprocess.py reads every .mid file it finds here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="Number of chorales to export")
    parser.add_argument("--out", type=str, default="data/midi", help="Output folder")
    args = parser.parse_args()
    main(args.n, args.out)
