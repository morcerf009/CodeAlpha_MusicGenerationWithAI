# AI Music Generation System

## Author
Muzzammil Abbas

## Tech Stack
* **Python** (Core Logic)
* **TensorFlow / Keras** (Deep Learning Framework)
* **music21** (MIDI Data Processing)
* **NumPy** (Numerical Analysis)
* **FluidSynth** (Audio Synthesis)

## Project Objective
This project implements a Deep Learning system using Long Short-Term Memory (LSTM) networks to generate original MIDI music. The model is trained on classical MIDI patterns to learn the temporal relationships between notes and chords, allowing it to compose new musical sequences autonomously.

## Technical Implementation

### 1. Data Parsing & Representation
Using the `music21` library, MIDI files are parsed into a sequence of notes and chords. Chords are converted into a string format of their normal order (e.g., '0.4.7') to treat them as unique vocabulary elements similar to single notes.

### 2. Neural Network Architecture
The model uses a Sequential LSTM architecture designed for sequence prediction:
* **Input Layer**: Accepts sequences of 50 musical events.
* **LSTM Layers**: Two layers of 256 units each to capture long-term musical dependencies.
* **Dropout**: 30% dropout to prevent overfitting during training.
* **Output Layer**: A Softmax layer that predicts the probability distribution for the next note/chord in the sequence.

### 3. Generative Process
The generation pipeline starts with a random 'seed' sequence from the training data. The model predicts the most likely next note, which is then fed back into the network to predict the following note, creating a generative loop. Finally, the sequence is converted back into a MIDI stream and synthesized into audio using FluidSynth.
