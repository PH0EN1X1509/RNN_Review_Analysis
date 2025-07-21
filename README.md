# IMDB Movie Review Sentiment Analysis

A deep learning-based sentiment analysis system that uses a Simple RNN (Recurrent Neural Network) to classify movie reviews as positive or negative. The model is trained on the IMDB dataset and deployed using Streamlit for user interaction.

## Project Architecture

### Model Architecture
- **Input Layer**: Embedding layer (10000 features, 128 dimensions)
- **Hidden Layer**: SimpleRNN with 128 units and ReLU activation
- **Output Layer**: Dense layer with sigmoid activation for binary classification
- **Sequence Length**: 500 tokens (padded/truncated)

### Technical Components
- **Neural Network Framework**: TensorFlow/Keras
- **Model Type**: Sequential with Simple RNN
- **Word Embedding**: Custom embedding layer (128-dimensional space)
- **Preprocessing**: 
  - Text tokenization
  - Sequence padding
  - Word index mapping
  - Vocabulary size: 10,000 most frequent words

## Requirements
```python
numpy
tensorflow
streamlit
```


## Model Training Details
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-entropy
- **Early Stopping**: Implemented with 5 epochs patience
- **Validation Split**: 20%
- **Batch Size**: 32
- **Maximum Epochs**: 10

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run main.py
```

3. Enter a movie review in the text area and click 'Classify' to get the sentiment prediction.

## Model Prediction Output
- **Sentiment**: Binary classification (Positive/Negative)
- **Prediction Score**: Probability score between 0 and 1
  - Score > 0.5: Positive sentiment
  - Score ≤ 0.5: Negative sentiment

## Implementation Details

### Text Preprocessing Pipeline
1. Text tokenization using IMDB word index
2. Sequence encoding with word-to-index mapping
3. Padding sequences to fixed length (500 tokens)
4. Embedding layer transformation

### Web Interface
Built using Streamlit with:
- Text input area for movie reviews
- Classification button
- Display for sentiment prediction and confidence score

## Performance Considerations
- Model weights are loaded once at startup
- Preprocessing is performed in real-time for each input
- Fixed sequence length of 500 tokens for consistent processing

## Technical Notes
- The model uses a pre-trained word index from the IMDB dataset
- Unknown words are handled with a special token
- Sequence padding ensures consistent input dimensions
- The model is saved in HDF5 format

