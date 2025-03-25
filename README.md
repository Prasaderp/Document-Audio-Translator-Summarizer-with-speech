# Audio and Document Processing System [Eng-Mar]

## Project Overview
The **Audio and Document Processing System** is a cutting-edge, professionally crafted solution designed to transform audio and document inputs into actionable, multilingual outputs with ease and precision. This system seamlessly extracts text from audio recordings and documents (PDF/DOCX), translates it into Marathi, generates concise summaries in both English and Marathi, and converts the Marathi summary into natural-sounding audio with adjustable playback speed. Built with a focus on performance, scalability, and user accessibility, it combines advanced AI models with intuitive interfaces to meet diverse professional needs.

## Core Features
- **Advanced Text Extraction**: Efficiently pulls text from audio using Whisper and from PDF/DOCX files with PyMuPDF and python-docx, ensuring high accuracy.
- **Seamless Translation**: Employs the NLLB-200-3.3B model to translate content into Marathi, enhanced with dynamic batching and caching for rapid processing.
- **Intelligent Summarization**: Leverages the LLaMA-3.1-8B-Instruct model to create concise English summaries, followed by Marathi translations for bilingual utility.
- **Natural Marathi Audio**: Generates lifelike Marathi speech from summaries using the MMS-TTS model, with speed customization via torchaudio for enhanced listening.
- **Dual Interface Design**: Provides a sleek Gradio web interface for interactive exploration and a robust Flask API for programmatic integration, supported by ngrok for global access.

## Technical Highlights
- **AI Models**: Integrates NLLB-200-3.3B (translation), LLaMA-3.1-8B-Instruct (summarization), Whisper (transcription), and MMS-TTS (speech synthesis).
- **Performance Optimization**: Features 4-bit quantization, adaptive batch sizing, and GPU/CPU memory management for efficient resource use.
- **Framework Support**: Built on Python 3.8+, PyTorch, Transformers, Gradio, Flask, and other industry-standard libraries.
- **Deployment Flexibility**: Runs locally or exposes a public URL via ngrok, catering to both standalone and networked environments.

## How to Use
1. **Input Submission**: Upload an audio file or document through the Gradio interface or send via the Flask API endpoint.
2. **Automated Processing**: The system handles extraction, translation, summarization, and audio generation in a streamlined workflow.
3. **Output Retrieval**: View results interactively in Gradio (text and audio) or receive structured JSON responses via the API.

## System Requirements
- **Hardware**: GPU (CUDA-enabled) recommended (min T4 GPU) for good performance.
- **Software**: Install dependencies with `pip install -r requirements.txt` (create based on script imports).
- **Authentication**: Requires a Hugging Face token for model access and ngrok auth, securely managed within the system.

## Licensing
Released under the **MIT License**, this project encourages professional use, adaptation, and collaboration while maintaining open-source integrity.

## Get in Touch
For support, feature requests, or collaboration opportunities, please file a GitHub issue or contact the development team via email (itsprasadsomvanshi@gmail.com).
