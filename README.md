# ColabBench

A collection of practical Google Colab notebooks demonstrating various AI, machine learning, and NLP techniques. This repository serves as a hands-on playground for exploring cutting-edge technologies in artificial intelligence.

## 📚 Notebooks Overview

### 🎯 Prompt Engineering & AI Agents

#### **Anthropic_Prompt_Generator.ipynb**
- **Purpose**: Interactive prompt engineering tool using Anthropic's "Metaprompt" technique
- **Key Features**: 
  - Automated prompt template generation for various tasks
  - Examples for customer service, document analysis, math tutoring, and function calling
  - Built-in testing framework for generated prompts
- **Technologies**: Anthropic Claude API, prompt engineering best practices
- **Use Case**: Solve the "blank page problem" when creating AI prompts

#### **CrewAI_Claude.ipynb**
- **Purpose**: Multi-agent AI system for automated blog post creation with human input
- **Key Features**:
  - Three specialized AI agents: Research Specialist, Content Writer, and File Archiver
  - Human-in-the-loop research topic selection
  - Automated web research using DuckDuckGo
  - Content generation and markdown file saving
- **Technologies**: CrewAI framework, Anthropic Claude Haiku, LangChain
- **Use Case**: Collaborative AI content creation with research and writing workflows

#### **Franc_v0_9_Sandbox.ipynb**
- **Purpose**: RAG (Retrieval-Augmented Generation) chatbot for running advice
- **Key Features**:
  - Conversational AI with memory using Llama 2 7B Chat model
  - Vector search using Pinecone for scientific literature
  - Specialized running coach persona with scientific backing
  - Streaming response capabilities
- **Technologies**: Hugging Face Transformers, Pinecone, LangChain, sentence-transformers
- **Use Case**: Domain-specific chatbot with scientific knowledge retrieval

### 🤖 Large Language Model Training & Fine-tuning

#### **QLoRA.ipynb**
- **Purpose**: Demonstrates Parameter-Efficient Fine-Tuning using QLoRA (Quantized LoRA)
- **Key Features**:
  - 4-bit quantization for memory-efficient training
  - Low-Rank Adaptation (LoRA) for efficient model customization
  - Reduces computational requirements while maintaining performance
- **Technologies**: QLoRA, LoRA, 4-bit quantization, Hugging Face
- **Use Case**: Fine-tune large language models on consumer hardware

#### **RAG_with_LLaMa_13B.ipynb**
- **Purpose**: Implementation of RAG system using LLaMA 13B model
- **Key Features**:
  - Large-scale language model integration
  - Document retrieval and generation pipeline
  - Combines parametric and non-parametric knowledge
- **Technologies**: LLaMA 13B, vector databases, retrieval systems
- **Use Case**: Build knowledge-grounded AI assistants

### 📄 Document Processing & Understanding

#### **Nougat.ipynb**
- **Purpose**: OCR and document parsing using Meta's Nougat model
- **Key Features**:
  - Academic paper PDF to markdown conversion
  - Mathematical equation recognition
  - Table and figure extraction
  - Scientific document structure understanding
- **Technologies**: Nougat (Neural Optical Understanding for Academic documents)
- **Use Case**: Convert academic PDFs to structured, searchable text

#### **Extractive_Summarization.ipynb**
- **Purpose**: Document summarization using extractive methods
- **Key Features**:
  - Sentence-level importance scoring
  - Key information extraction from long documents
  - Maintains original text integrity
- **Technologies**: Natural language processing, text ranking algorithms
- **Use Case**: Automated document summarization and key point extraction

#### **Fine_tune_Longformer_Encoder_Decoder_(LED)_for_Summarization_on_pubmed.ipynb**
- **Purpose**: Fine-tune Longformer-Encoder-Decoder for scientific paper summarization
- **Key Features**:
  - Long document processing (up to 16K tokens)
  - PubMed dataset training for medical/scientific text
  - Encoder-decoder architecture for abstractive summarization
- **Technologies**: Longformer-LED, Hugging Face Transformers, PubMed dataset
- **Use Case**: Specialized summarization for scientific literature

### 🏗️ Machine Learning Infrastructure

#### **Ludwig_+_DeepLearning_ai_Hands_On_Workshop.ipynb**
- **Purpose**: Hands-on workshop demonstrating Ludwig's declarative ML framework
- **Key Features**:
  - No-code/low-code machine learning
  - Declarative model configuration
  - Automated feature engineering and model selection
  - Integration with deep learning workflows
- **Technologies**: Ludwig framework, automated ML, declarative configuration
- **Use Case**: Rapid prototyping and deployment of ML models without extensive coding

## 🚀 Getting Started

### Prerequisites
- Google Colab account
- API keys for respective services (Anthropic, Hugging Face, Pinecone, etc.)
- Basic understanding of Python and machine learning concepts

### Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/lrtherond/ColabBench.git
   ```

2. **Open notebooks in Google Colab**:
   - Upload notebooks to your Google Drive
   - Open with Google Colab
   - Install required dependencies (usually handled in first cells)

3. **Configure API Keys**:
   - Use Google Colab's `userdata.get()` for secure API key management
   - Or set environment variables in notebook cells

## 🛠️ Technologies Used

- **AI Frameworks**: LangChain, CrewAI, Ludwig
- **Models**: Claude (Anthropic), LLaMA, Longformer, various Hugging Face models
- **Vector Databases**: Pinecone
- **ML Libraries**: Hugging Face Transformers, sentence-transformers
- **Document Processing**: Nougat, various NLP libraries
- **Development**: Google Colab, Python, Jupyter notebooks

## 📈 Use Cases

- **Research & Development**: Experiment with latest AI/ML techniques
- **Prototyping**: Rapid development of AI-powered applications
- **Learning**: Hands-on experience with cutting-edge technologies
- **Production**: Scalable patterns for real-world deployment

## 🤝 Contributing

Feel free to contribute by:
- Adding new notebook examples
- Improving existing implementations
- Fixing bugs or updating dependencies
- Sharing interesting use cases

## 📝 License

This collection is provided for educational and research purposes. Please respect the individual licenses of the technologies and models used.

## ⚠️ Important Notes

- **API Costs**: Some notebooks use paid APIs (Claude, OpenAI, etc.) - monitor usage
- **Computational Requirements**: Some models require significant GPU resources
- **Data Privacy**: Be cautious with sensitive data when using cloud-based models
- **Model Versions**: Models and APIs evolve rapidly - check for updates regularly

---

*This repository serves as a comprehensive playground for exploring modern AI technologies through practical, hands-on examples.*