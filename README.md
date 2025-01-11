# TradingStrategyLearner

An advanced machine learning system that learns trading strategies and concepts from educational trading videos and converts them into executable trading strategies.

## Features

- **Video Processing**: Extract frames and analyze visual content from financial videos
- **Audio Processing**: Process audio content for speech recognition and analysis
- **Text Detection**: Extract text from video frames using OCR
- **Knowledge Extraction**: Generate insights from extracted text and audio content
- **Strategy Generation**: Create trading strategies based on extracted knowledge
- **Visualization**: Generate comprehensive visualizations of analysis results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AshutoshBuilds/TradingStrategyLearner.git
cd TradingStrategyLearner
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install required model files:
```bash
# Install spaCy models
python -m spacy download en_core_web_sm

# Download other required models
mkdir -p models
cd models

# Whisper model for speech recognition
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -O whisper-medium.pt

# LLaVA model for vision-language tasks
wget https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/main/pytorch_model.bin -O llava-model.bin

# Sentence Transformers
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin -O sentence-transformer.bin

# Emotion Classification
wget https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/resolve/main/pytorch_model.bin -O emotion-classifier.bin
```

5. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg

# macOS
brew install tesseract ffmpeg

# Windows
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install FFmpeg from: https://www.ffmpeg.org/download.html
```

## Required Model Downloads

Before running the system, you need to download and set up the following models:

1. Core Models:
```bash
# Create models directory if it doesn't exist
mkdir -p models

# Whisper Model (Speech Recognition)
mkdir -p models/whisper
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -O models/whisper/base.pt

# Emotion Classification Model
mkdir -p models/emotion_classifier
huggingface-cli download j-hartmann/emotion-english-distilroberta-base --local-dir models/emotion_classifier

# Sentence Transformers (downloaded automatically on first run)
# Location: models/sentence_transformers/all-MiniLM-L6-v2

# spaCy Models
python -m spacy download en_core_web_lg
```

Note: The LLaVA model (Visual Language) is already included in the `models/concept_model` directory.

2. Trading-Specific Models:
```bash
# Pattern Detection Models
mkdir -p models/pattern_detection
wget [URL] -O models/pattern_detection/model.pt

# Indicator Detection Models
mkdir -p models/indicator_detection
wget [URL] -O models/indicator_detection/model.pt

# Signal Generation Models
mkdir -p models/signal_generation
wget [URL] -O models/signal_generation/model.pt

# Strategy Generation Models
mkdir -p models/strategy_generation
wget [URL] -O models/strategy_generation/model.pt
```

Note: Replace [URL] with actual model URLs. Some models will be downloaded automatically on first run.

## Model Directory Structure

```
models/
├── whisper/                  # Speech recognition
│   └── base.pt
├── sentence_transformers/    # Text embeddings
│   └── all-MiniLM-L6-v2/
├── emotion_classifier/       # Emotion analysis
├── concept_model/           # Trading concept recognition
├── text_overlay_model/      # OCR and text detection
├── strategy_model/          # Strategy generation
├── component_generation/    # Component analysis
├── temporal_flow/           # Time series analysis
├── indicator_detection/     # Technical indicator detection
├── pattern_detection/       # Chart pattern recognition
├── signal_generation/       # Trading signal generation
├── strategy_generation/     # Strategy composition
└── knowledge_base.json      # Trading knowledge base
```

## Usage

1. Basic usage:
```python
from src.main import TradingStrategyLearner

# Initialize the learner
learner = TradingStrategyLearner()

# Process a video
results = learner.process_video("path/to/your/video.mp4")

# Results will be saved to the output directory
print(f"Results saved to: {results['output_dir']}")
```

2. Using custom configuration:
```python
# Create a config file (config.json)
{
    "video_processor": {
        "frame_rate": 2,
        "resize_width": 1920,
        "resize_height": 1080
    },
    "output": {
        "save_intermediates": true,
        "output_dir": "custom_output"
    }
}

# Initialize with custom config
learner = TradingStrategyLearner(config_path="config.json")
```

## Project Structure

```
TradingStrategyLearner/
├── src/
│   ├── content_processor/
│   │   ├── video_processor.py
│   │   ├── audio_processor.py
│   │   ├── text_detector.py
│   │   └── knowledge_extractor.py
│   ├── strategy/
│   │   └── strategy_generator.py
│   ├── visualization/
│   │   └── visualizer.py
│   └── main.py
├── models/
│   ├── whisper-medium.pt
│   ├── llava-model.bin
│   ├── sentence-transformer.bin
│   └── emotion-classifier.bin
├── tests/
│   └── ...
├── output/
│   └── ...
├── requirements.txt
├── config.json
└── README.md
```

## Configuration

The system can be configured using a JSON configuration file. Available options:

```json
{
    "video_processor": {
        "frame_rate": 1,
        "resize_width": 1280,
        "resize_height": 720
    },
    "audio_processor": {
        "sample_rate": 16000,
        "chunk_duration": 30
    },
    "text_detector": {
        "min_confidence": 0.5
    },
    "knowledge_extractor": {
        "model_path": "en_core_web_sm",
        "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "embedding_model": "all-MiniLM-L6-v2"
    },
    "strategy_generator": {
        "llm_model": "gpt2",
        "embedding_model": "all-MiniLM-L6-v2"
    },
    "visualizer": {
        "style": "darkgrid",
        "figsize": [12, 8]
    },
    "output": {
        "save_intermediates": true,
        "output_dir": "output"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "trading_learner.log"
    }
}
```

## Output Format

The system generates the following outputs:

1. Extracted Knowledge:
   - Named entities
   - Key phrases
   - Sentiment analysis
   - Topic modeling
   - Content summary
   - Relationship extraction

2. Market Data:
   - Price analysis
   - Volume analysis
   - Technical indicators
   - Market conditions

3. Trading Strategy:
   - Strategy template
   - Parameters
   - Entry/exit conditions
   - Performance metrics
   - Confidence score

4. Visualizations:
   - Price analysis charts
   - Sentiment analysis plots
   - Topic distribution
   - Strategy evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Ashutosh Shukla
Email: ashutoshshukla734.as@gmail.com

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for transformer models
- spaCy for NLP capabilities
- PyTorch community for deep learning tools 