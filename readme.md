# AI Music Generation

## Author
Muzzammil Abbas

## Tech Stack
- **Python**
- **TensorFlow / Keras**
- **music21**
- **NumPy**
- **FluidSynth / midi2audio**

## Objective
The objective of this project is to implement a Deep Learning system capable of generating original MIDI music. By leveraging Long Short-Term Memory (LSTM) networks, the model learns the patterns, rhythms, and harmonic structures found in classical MIDI files to compose new sequences of music.

## How it Works

### 1. Data Parsing
Using the `music21` library, MIDI files are parsed to extract individual notes and chords. Chords are converted into a string representation of their normal order to allow the model to treat them as unique entities similar to single notes.

### 2. Data Preparation
- **Mapping**: All unique notes and chords are mapped to integer values to create a numerical vocabulary.
- **Sequencing**: The data is processed into sequences of length 50. Each input sequence is associated with a single output (the next note or chord in the sequence).
- **Normalization**: Input sequences are normalized by the vocabulary size, and output labels are one-hot encoded.

### 3. Model Architecture
The project utilizes a Sequential model consisting of:
- **LSTM Layers**: To capture long-term dependencies and temporal patterns in musical sequences.
- **Dropout Layers**: To prevent overfitting during training.
- **Dense Layers**: A fully connected layer with ReLU activation followed by a Softmax output layer to predict the probability distribution over the musical vocabulary.

### 4. Generation Pipeline
To generate music, a random starting seed from the input data is provided to the model. The model predicts the next note, which is then appended to the sequence to predict the subsequent note, creating a generative loop. The resulting integer sequence is then converted back into MIDI format using `music21`.
