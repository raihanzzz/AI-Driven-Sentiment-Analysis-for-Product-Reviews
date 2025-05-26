# AI-Driven-Sentiment-Analysis-for-Product-Reviews

This project applies **Natural Language Processing (NLP)** and **Machine Learning/Deep Learning techniques** to perform **sentiment classification** on Amazon product reviews. It is designed to evaluate the sentiment (positive or negative) of user reviews using various models, including **Naive Bayes**, **SVM**, **LSTM**, and **BERT**.

## 🔍 Project Objective

The primary goal is to address the **cross-domain sentiment classification** challenge: how to build models that maintain accuracy even when trained on one product category and tested on another.

## 💡 Key Features

- Preprocessing pipeline using NLTK, spaCy
- Feature extraction using TF-IDF, Word2Vec, GloVe, and BERT embeddings
- Model training using:
  - Traditional ML: Naive Bayes, SVM
  - Deep Learning: LSTM
  - Transformers: BERT (via HuggingFace)
- Evaluation using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve
- Performance comparison across same-domain and cross-domain scenarios

## 📁 Dataset

- **Blitzer's Amazon Review Dataset**
  - Domains: Books, DVDs, Electronics, Kitchen, etc.
  - Includes labeled reviews for binary sentiment analysis (positive/negative)

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Frameworks & Libraries:**
  - NLP: NLTK, spaCy
  - ML/DL: Scikit-learn, TensorFlow/Keras, PyTorch
  - Transformers: HuggingFace Transformers
  - Visualization: Matplotlib, Seaborn
- **Development Tools:** Google Colab, Jupyter Notebook, VS Code

## 📊 Model Performance

| Model        | Same-Domain Accuracy | Cross-Domain Accuracy |
|--------------|----------------------|------------------------|
| Naive Bayes  | 78.2%                | 64.3%                  |
| SVM          | 86.1%                | 72.5%                  |
| LSTM         | 88.7%                | 75.4%                  |
| BERT         | **92.3%**            | **85.1%**              |

## 📌 Project Structure

```
📁 sentiment-analysis/
├── data/                  # Amazon review datasets
├── notebooks/             # Jupyter/Colab notebooks
├── models/                # Trained models (optional)
├── utils/                 # Preprocessing and helper scripts
├── results/               # Visualizations and output logs
└── README.md
```

## 🔬 Future Enhancements

- Real-time web interface for sentiment prediction
- Aspect-based sentiment analysis
- Multilingual support using XLM-R or mBERT
- Integration of Explainable AI (XAI) for model interpretability
- Optimization using lightweight transformer models (DistilBERT, TinyBERT)

## 👥 Contributors

- **Arhamuddin** – Data preprocessing, traditional ML modeling
- **Md Raihan** – Deep learning modeling, literature review, results analysis
- **Md Adnan Nasim** – BERT integration, system design
- **Tabish Iqbal** – Report formatting, code testing, research on future scope

## 📄 License

This project is for academic use only and is part of the **Skill X Project** under **JIS University**.
