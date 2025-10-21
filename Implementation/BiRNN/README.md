# Bidirectional RNN (Bi-RNN) Implementation

Implementation of Bidirectional Recurrent Neural Networks for enhanced sequence understanding.

## Overview

Bidirectional RNNs process sequences in both forward and backward directions, providing complete context for better predictions. This is particularly useful when the entire sequence is available before making predictions.

## Architecture

```
Input Sequence
    ↓
Embedding Layer
    ↓
    ├─→ Forward RNN ──→┐
    └─→ Backward RNN ─→┤
                        ↓
                  Concatenate
                        ↓
                   Dense Layers
                        ↓
                      Output
```

## Key Advantages

1. **Bidirectional Context**: Processes sequence both ways
2. **Better Accuracy**: Typically 2-5% improvement over unidirectional
3. **Complete Information**: Uses full sequence context
4. **Improved Understanding**: Captures dependencies from both directions

## When to Use

- Text classification (sentiment, topic)
- Named entity recognition
- Part-of-speech tagging
- Speech recognition (offline)
- Protein structure prediction

## Performance

- **Accuracy**: ~85-90% on IMDB
- **Parameters**: 2x unidirectional RNN
- **Training Time**: ~1.5-2x longer

## Installation & Usage

```bash
cd Implementation/BiRNN
jupyter notebook BiRNN.ipynb
```

## Author

Pravin Kumar S
