# Generative AI: Text Generation Models Comparative Analysis

This project explores and compares text generation models trained on different datasets: OpenWebText and Gutenberg Digital Books. By examining how the same model architecture performs on these distinct data distributions, we gain valuable insights into both model capabilities and dataset characteristics.

## 🔍 Project Overview

This project implements and analyzes GPT-2 language models across two distinct text corpora:

1. **OpenWebText** - A collection of web content that mimics the training data of GPT-2, containing modern, diverse content
2. **Gutenberg Digital Books** - Classic literature from Project Gutenberg, featuring formal language and literary style

Through this comparative analysis, we explore:
- How dataset characteristics influence model generation style
- Domain-specific adaptations in language models
- Evaluation metrics across different text domains
- Insights into underlying data distributions through generative modeling

## 📊 Key Components

The Jupyter notebook (`text_generation_training.ipynb`) contains:

### 1. Model Implementation
- GPT-2 architecture using the Transformers library
- Custom dataset preprocessing for both text corpora
- Fine-tuning procedure with mixed precision training

### 2. Comparative Analysis
- Training dynamics visualization
- Cross-domain performance evaluation
- Detailed text characteristic analysis
- Domain classification using perplexity scores

### 3. Creative Applications
- Text generation with parameter exploration
- Style transfer between domains
- Domain classification system

## 🧪 Methodology

Our approach follows these key steps:

1. **Dataset Preparation**: Load and preprocess samples from OpenWebText and Gutenberg
2. **Model Training**: Fine-tune DistilGPT-2 on each dataset separately
3. **Text Generation**: Generate text using both models with various parameters
4. **Evaluation**: Calculate perplexity and analyze text characteristics
5. **Comparative Analysis**: Directly compare model outputs and behaviors
6. **Applications**: Demonstrate practical applications like style transfer

## 📈 Key Findings

Our analysis reveals:

1. **Dataset Influence**: Training data profoundly shapes generation style:
   - OpenWebText model produces contemporary, factual-sounding text
   - Gutenberg model generates formal, literary text with archaic language

2. **Domain Adaptation**: Each model performs better on texts from its own domain

3. **Quantifiable Differences**: Clear patterns in:
   - Lexical diversity (higher in Gutenberg model)
   - Sentence complexity (more elaborate in Gutenberg model)
   - Word choice and vocabulary distribution
   - Temperature sensitivity

4. **Practical Applications**: Different models excel in different generation tasks depending on the desired style and content

## 🔧 Technical Details

The implementation uses:
- PyTorch for model training
- Transformers library for GPT-2 implementation
- Mixed precision training for efficiency
- NVIDIA CUDA for GPU acceleration (when available)
- Perplexity as primary evaluation metric
- Custom text analysis for style comparison

## 🚀 Getting Started

1. Install the required dependencies:
   \`\`\`bash
   pip install torch transformers datasets tqdm matplotlib nltk wordcloud scikit-learn pandas seaborn streamlit
   \`\`\`

2. Run the Jupyter notebook to train models and see the analysis:
   \`\`\`bash
   jupyter notebook text_generation_training.ipynb
   \`\`\`

3. Launch the Streamlit app for interactive text generation:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## 🎯 Future Directions

Promising areas for extending this work include:
- Training larger models on more comprehensive datasets
- Multi-domain training for versatile generation
- More sophisticated style transfer techniques
- Human evaluation studies for qualitative assessment
- Ethical analysis of domain-specific generation

## 📚 References

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners.
2. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing.
3. Gokaslan, A., & Cohen, V. (2019). OpenWebText Corpus.
4. Project Gutenberg. https://www.gutenberg.org/
5. Holtzman, A., et al. (2019). The curious case of neural text degeneration.
