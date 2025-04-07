# Jazz Piano Music Generation Using LSTM

![Music Generation](https://img.shields.io/badge/Genre-Classical_Piano-blue)
![Deep Learning](https://img.shields.io/badge/Model-LSTM-orange)
![Python](https://img.shields.io/badge/Python-3.x-green)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Project Overview
This project implements an LSTM-based neural network for generating classical piano music from MIDI files. The model learns musical patterns and structures to create original compositions that mimic classical piano style.

## Key Features
- MIDI file processing pipeline using music21 library
- Customizable LSTM architecture with dropout layers
- Advanced music representation using note/chord encoding
- 10-fold cross-validation for robust evaluation
- Comprehensive metrics including KL divergence and cosine similarity

## Dataset
The model was trained on:
- **Training Set**: 20 jazz piano MIDI files (`jazz_sample_files/`)
- **Validation Set**: 20 Separate MIDI samples for evaluation (`Validation_samples/`)

Dataset Statistics:
- Total notes processed: 24,483
- Unique notes after preprocessing: 548
- Sequence length: 40 notes

## Model Architecture
```python
Sequential([
    LSTM(256, input_shape=(40, 1), return_sequences=True),
    Dropout(0.1),
    LSTM(128),
    Dense(128),
    Dropout(0.1),
    Dense(vocab_size, activation='softmax')
])
```
- Optimizer: Adamax (learning_rate=0.01)
- Loss: Categorical Crossentropy
- Batch Size: 256

## Evaluation Metrics
The model was evaluated using three key metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| KL Divergence | Measures difference between real and generated note distributions | Lower is better |
| Interval Similarity | Cosine similarity of note intervals | Higher is better |
| Rhythm Similarity | Cosine similarity of rhythmic patterns | Higher is better |

## Usage
1. Prepare your MIDI files:
   - Place training files in `jazz_sample_files/`
   - Place validation files in `Validation_samples/`

2. Run the Jupyter notebook:
```bash
jupyter notebook Music_Generation_LSTM.ipynb
```

3. To generate new music:
```python
seed = random.choice(features)
generated_music = generate_music(model, seed, length=100)
```

## Results
After 10-fold cross-validation:
- Average KL Divergence: 1.24
- Average Interval Similarity: 0.82
- Average Rhythm Similarity: 0.78

Sample generated music can be saved as MIDI files using the provided utilities.

## Future Work
- Incorporate velocity and dynamics for more expressive music
- Experiment with transformer architectures
- Add conditional generation based on mood/emotion
- Develop a web interface for interactive music generation

---

*Note: Replace placeholder values with your actual results and GitHub URLs*
