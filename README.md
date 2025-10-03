# Urdu-Roman Urdu Transliterator

A BiLSTM-based neural network application for transliterating Urdu text to Roman Urdu, deployed using Streamlit.

## Features

- Real-time Urdu to Roman Urdu transliteration
- Interactive web interface with beautiful UI
- Translation history tracking
- Performance metrics dashboard
- Batch processing capabilities
- Copy-to-clipboard functionality

## Model Architecture

- BiLSTM (Bidirectional Long Short-Term Memory) neural network
- Attention mechanism for improved accuracy
- Character-level and word-level transliteration support

## Deployment

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/hurairamuzammal/NLP_A01.git
cd NLP_A01
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Single Text Transliteration**:
   - Enter Urdu text in the input area
   - Click "Transliterate" button
   - View the Roman Urdu output
   - Copy the result to clipboard

2. **Quick Examples**:
   - Use predefined examples from the sidebar
   - Click any example to auto-fill the input

3. **History Tracking**:
   - View recent transliterations in the sidebar
   - Track total number of transliterations

## License

This project is for educational purposes as part of NLP coursework.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.