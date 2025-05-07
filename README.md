# AI Earnings Call Analyzer

<div align="center">

# Nonanswer Detection in Earnings Call Transcripts

**Advanced AI Solution for Financial Communication Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![GPT-2](https://img.shields.io/badge/Model-GPT--2-green.svg)](https://openai.com/research/gpt-2)
[![LLaMA 3.1](https://img.shields.io/badge/Model-LLaMA%203.1-orange.svg)](https://ai.meta.com/llama/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Winner of Gen AI Visionaries 2025 Competition*

</div>

## 📋 Executive Summary

This project presents an innovative AI solution designed to identify evasive responses in earnings call transcripts. By leveraging state-of-the-art language models and advanced NLP techniques, we provide financial analysts with a powerful tool for detecting nonanswers in corporate communications.

## 🎯 Problem Statement

Financial analysts face significant challenges in analyzing earnings call transcripts:
- Traditional NLP systems struggle with varied speaking styles and context
- Manual analysis is time-consuming and resource-intensive
- Lack of standardized methods for identifying evasive responses

## 💡 Solution

Our solution combines multiple advanced AI technologies:
- **Fine-tuned GPT-2 Model**: Primary classification engine
- **LLaMA 3.1**: Advanced reasoning and explanation generation
- **Streamlit Interface**: User-friendly deployment platform
- **Groq API**: High-performance inference engine

## 👥 Development Team

- Syed Shahbuddin
- Utsav Soni
- Siddesh Sharma
- Nithin G

## 📊 Technical Specifications

### Dataset Composition
- **Total Samples**: 1,740
- **Class Distribution**:
  - Answers: 903 (51.9%)
  - Nonanswers: 834 (48.1%)
- **Data Enhancement**: Synthetic samples generated via LLMs for balanced training

### System Architecture

<div align="center">
  <img src="assets/architecture.png" alt="Solution Architecture" width="700">
</div>

#### Core Components

1. **Data Ingestion**
   - Raw earnings call transcript processing
   - Automated text extraction and formatting

2. **Preprocessing Pipeline**
   - Text normalization and cleaning
   - Tokenization and feature extraction
   - Binary classification mapping

3. **Model Architecture**
   - GPT-2 fine-tuning for response classification
   - LLaMA 3.1 integration for explainability
   - Confidence scoring and validation

4. **Deployment Stack**
   - Streamlit-based user interface
   - Groq API for optimized inference
   - Scalable cloud infrastructure

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Pip package manager
- 4GB RAM minimum
- Internet connection for API access

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/syed-shahbuddin/AI-EarningsCallAnalyzer.git
   cd AI-EarningsCallAnalyzer
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Application**
   ```bash
   streamlit run app.py
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

