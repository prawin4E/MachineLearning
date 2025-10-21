# Recurrent Neural Network (RNN) Implementation

A comprehensive implementation of Recurrent Neural Networks for sequence prediction and sentiment analysis.

## Overview

This project demonstrates how to build, train, and evaluate a Recurrent Neural Network using TensorFlow and Keras. RNNs are designed to work with sequential data by maintaining an internal state (memory).

## What is a Recurrent Neural Network?

A **Recurrent Neural Network (RNN)** is a type of neural network designed for processing sequential data. Unlike feedforward networks, RNNs have connections that form cycles, allowing information to persist across time steps.

### Key Components:

1. **Hidden State**: Memory that persists across time steps
2. **Recurrent Connections**: Feedback loops in the network
3. **Embedding Layer**: Converts discrete tokens to continuous vectors
4. **Time Steps**: Sequential processing of input
5. **Unfolding**: RNN viewed as a deep network across time

## Architecture

```
Input Sequence (200 time steps)
    ↓
Embedding (vocab_size → 128 dimensions)
    ↓
SimpleRNN (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
SimpleRNN (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.3)
    ↓
Output (1, Sigmoid) → Binary Classification
```

## Topics Covered

1. **Sequential Data Processing**
   - Text tokenization
   - Sequence padding
   - Vocabulary management

2. **RNN Architecture**
   - SimpleRNN layers
   - Embedding layers
   - Hidden state management

3. **Sentiment Analysis**
   - Binary classification
   - Text preprocessing
   - Model evaluation

4. **Training Techniques**
   - Early stopping
   - Dropout regularization
   - Batch processing

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage

```bash
cd Implementation/RNN
jupyter notebook RNN.ipynb
```

## Dataset

### IMDB Movie Reviews (Default)
- **Reviews**: 25,000 training + 25,000 test
- **Classes**: 2 (positive/negative sentiment)
- **Vocabulary**: 10,000 most frequent words
- **Max Length**: 200 words per review
- **Task**: Binary sentiment classification

## Model Performance

Expected performance on IMDB:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~80-85%
- **Training Time**: 10-15 minutes (CPU)

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary Size | 10,000 | Number of unique words |
| Max Sequence Length | 200 | Maximum words per review |
| Embedding Dim | 128 | Word vector dimensions |
| RNN Units | 128, 64 | Hidden units per layer |
| Dropout Rate | 0.3 | Regularization |
| Batch Size | 128 | Samples per update |
| Epochs | 10 | Training iterations |

## Key Concepts

### 1. Recurrent Connections
```
h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b)
```
- **h_t**: Hidden state at time t
- **x_t**: Input at time t
- **W_hh**: Recurrent weights
- **W_xh**: Input weights

### 2. Backpropagation Through Time (BPTT)
- Unrolls RNN across time steps
- Computes gradients through entire sequence
- Can suffer from vanishing/exploding gradients

### 3. Embedding Layer
- Maps discrete words to continuous vectors
- Learns word representations during training
- Reduces dimensionality
- Captures semantic relationships

### 4. Sequence Padding
- Makes all sequences same length
- **Post-padding**: Adds zeros at end
- **Pre-padding**: Adds zeros at start
- Required for batch processing

## When to Use RNN

### Perfect For:
- Short text classification
- Sentiment analysis
- Name entity recognition
- Simple time series
- Speech recognition (basic)

### Not Ideal For:
- Long sequences (use LSTM/GRU)
- Images (use CNN)
- Tabular data (use ANN)
- Very long-term dependencies

## Advantages

1. **Sequential Processing**: Handles variable-length sequences
2. **Parameter Sharing**: Same weights across time steps
3. **Memory**: Maintains information across time
4. **Flexible**: Works with different input/output lengths

## Limitations

1. **Vanishing Gradients**: Struggles with long sequences
2. **Sequential Computation**: Cannot parallelize easily
3. **Short-term Memory**: Forgets early information
4. **Training Time**: Slower than feedforward networks

## Common RNN Variants

### 1. SimpleRNN (Vanilla RNN)
- Basic recurrent unit
- Suffers from vanishing gradients
- Good for short sequences

### 2. LSTM (Long Short-Term Memory)
- Solves vanishing gradient problem
- Better long-term memory
- More parameters

### 3. GRU (Gated Recurrent Unit)
- Simplified LSTM
- Fewer parameters
- Similar performance

### 4. Bidirectional RNN
- Processes sequence both directions
- Better context understanding
- More computation

## Best Practices

1. **Gradient Clipping**: Prevent exploding gradients
2. **Use LSTM/GRU**: For longer sequences
3. **Embedding Layer**: For text data
4. **Dropout**: Prevent overfitting
5. **Batch Normalization**: Stabilize training
6. **Shorter Sequences**: Truncate if possible

## RNN vs Other Architectures

| Feature | RNN | LSTM | CNN | Transformer |
|---------|-----|------|-----|-------------|
| Sequential | Yes | Yes | No | No |
| Long-term Memory | Poor | Excellent | N/A | Excellent |
| Parallelization | Poor | Poor | Excellent | Good |
| Training Speed | Slow | Slower | Fast | Medium |
| Best For | Short sequences | Long sequences | Images | Long text |

## Common Issues and Solutions

### Problem: Vanishing Gradients
**Solutions**:
- Use LSTM or GRU instead
- Reduce sequence length
- Use gradient clipping
- Initialize weights carefully

### Problem: Exploding Gradients
**Solutions**:
- Apply gradient clipping
- Reduce learning rate
- Use batch normalization
- Check weight initialization

### Problem: Overfitting
**Solutions**:
- Add dropout layers
- Use regularization
- Get more training data
- Reduce model complexity

### Problem: Slow Training
**Solutions**:
- Reduce sequence length
- Use smaller batches
- Simplify architecture
- Use GPU acceleration

## Applications

### Text Processing:
- Sentiment analysis
- Text classification
- Language modeling
- Machine translation

### Time Series:
- Stock price prediction
- Weather forecasting
- Sales forecasting
- Anomaly detection

### Speech:
- Speech recognition
- Speaker identification
- Emotion detection

## File Structure

```
RNN/
├── RNN.ipynb                  # Main Jupyter notebook
├── README.md                  # This file
└── rnn_sentiment_model.h5     # Saved model (generated)
```

## Future Improvements

- [ ] Implement LSTM comparison
- [ ] Add attention mechanism
- [ ] Try GRU architecture
- [ ] Implement bidirectional RNN
- [ ] Add different datasets
- [ ] Implement sequence-to-sequence
- [ ] Add text generation example

## Learning Resources

### Documentation
- [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Keras RNN Layers](https://keras.io/api/layers/recurrent_layers/)

### Papers
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Learning Long-Term Dependencies](https://www.bioinf.jku.at/publications/older/2604.pdf)

### Tutorials
- [Andrej Karpathy's RNN Tutorial](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Colah's Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Videos
- [StatQuest: RNNs](https://www.youtube.com/watch?v=AsNTP8Kwu80)
- [MIT Deep Learning: RNNs](https://www.youtube.com/watch?v=SEnXr6v2ifU)

## Contributing

Contributions welcome! Feel free to:
- Add new RNN variants
- Improve documentation
- Add more applications
- Optimize performance

## License

Part of the Machine Learning repository - MIT License

## Author

Pravin Kumar S

## Acknowledgments

- TensorFlow and Keras teams
- IMDB dataset providers
- NLP research community
