import glob
import numpy as np
import os
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from midi2audio import FluidSynth

def load_midi_dataset(path):
    notes = []
    for file in glob.glob(path):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def prepare_sequences(notes, n_vocab, sequence_length=50):
    pitchnames = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length):
        network_input.append([note_to_int[char] for char in notes[i:i + sequence_length]])
        network_output.append(note_to_int[notes[i + sequence_length]])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    network_output = to_categorical(network_output)
    return network_input, network_output, pitchnames

def create_model(input_shape, n_vocab):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

if __name__ == '__main__':
    print('Loading dataset...')
    raw_notes = load_midi_dataset('/content/*.mid')
    vocab_size = len(set(raw_notes))
    X, y, pitch_labels = prepare_sequences(raw_notes, vocab_size)
    
    print('Initializing and training model...')
    model = create_model((X.shape[1], X.shape[2]), vocab_size)
    model.fit(X, y, epochs=5, batch_size=64, verbose=1)
    
    print('Generating music...')
    int_to_note = {i: n for i, n in enumerate(pitch_labels)}
    pattern = X[np.random.randint(0, len(X)-1)]
    prediction_output = []
    for _ in range(100):
        pred_input = np.reshape(pattern, (1, len(pattern), 1))
        idx = np.argmax(model.predict(pred_input, verbose=0))
        prediction_output.append(int_to_note[idx])
        pattern = np.append(pattern, idx / float(vocab_size))[1:]

    offset = 0
    final_notes = []
    for p in prediction_output:
        if ('.' in p) or p.isdigit():
            final_notes.append(chord.Chord([note.Note(int(n)) for n in p.split('.')]))
        else:
            final_notes.append(note.Note(p))
        final_notes[-1].offset = offset
        offset += 0.5
    
    stream.Stream(final_notes).write('midi', fp='final_composition.mid')
    print('Saved: final_composition.mid')

!ls -l music_generator.py
