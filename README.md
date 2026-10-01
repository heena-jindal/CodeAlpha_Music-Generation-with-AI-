# 🎵 Music Generation with AI

Generates new short pieces of music by training a stacked LSTM on real
classical MIDI data, then sampling new note sequences from it and writing
them back out as a playable `.mid` file. **This full pipeline has been run
and verified end-to-end** (data → preprocessing → training → generation →
MIDI output).

## Pipeline
| Step | Script | What it does |
|---|---|---|
| 1. Collect data | `collect_data.py` | Exports real public-domain Bach chorales bundled with `music21`'s built-in corpus to `.mid` files — no external download needed. Drop in your own jazz/pop/classical `.mid` files here too. |
| 2. Preprocess | `preprocess.py` | Parses every MIDI file with `music21`, extracts a flat sequence of notes & chords, builds a vocabulary, and encodes fixed-length input→next-note training sequences. |
| 3. Build & train | `train_model.py` | Builds a 2-layer LSTM (with dropout + dense softmax head) in TensorFlow/Keras and trains it to predict the next note given a sequence of previous notes. |
| 4. Generate | `generate_music.py` | Seeds the trained model with a real sequence, autoregressively samples new notes/chords (temperature-controlled), and converts the result back into a MIDI file with `music21`. |

## Quickstart
```bash
pip install -r requirements.txt

python collect_data.py --n 40            # step 1: export 40 Bach chorales as MIDI
python preprocess.py --seq_len 50        # step 2: build training sequences
python train_model.py --epochs 100       # step 3: train the LSTM (takes longer with more epochs/data)
python generate_music.py --n_notes 200   # step 4: generate a new piece -> output/generated.mid
```
Then open `output/generated.mid` in any MIDI player (VLC, GarageBand,
QuickTime, an online player like https://onlinesequencer.net, or import it
into a DAW) to listen to or convert it to audio.

## Notes on quality vs. training time
- The **3-epoch smoke test** used during development produces musically
  weak (near-random) output — enough to prove the pipeline works end to end.
- For music that actually sounds structured, train for **100–200 epochs**
  on a reasonably sized dataset (100+ MIDI files). This can take from
  30 minutes (small dataset, CPU) to a few hours; a GPU speeds this up a lot.
- `--seq_len` (notes of context) and `--temperature` (randomness of
  generation) are the two easiest knobs to experiment with.

## Using your own dataset (jazz, pop, etc.)
Skip `collect_data.py` and just drop your own `.mid`/`.midi` files into
`data/midi/` — `preprocess.py` reads every MIDI file in that folder,
regardless of genre. Good free sources: the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/),
[Classical Archives](https://www.classicalarchives.com/), or your own
exported MIDI from a DAW.

## File structure
```
task3-music-generation/
├── collect_data.py     # Step 1: gather MIDI data (music21 corpus)
├── preprocess.py        # Step 2: MIDI -> note sequences (music21)
├── train_model.py        # Step 3: build & train the LSTM (TensorFlow/Keras)
├── generate_music.py     # Step 4-5: generate notes & write output MIDI
├── requirements.txt
├── .gitignore            # data/, models/, output/ are generated, not committed
└── README.md
```
`data/`, `models/`, and `output/` are created by the scripts and left out of
git (see `.gitignore`) since they're large/regeneratable — run the
quickstart commands above to recreate them locally.

## License
MIT for the code. The bundled Bach chorales are public domain.
