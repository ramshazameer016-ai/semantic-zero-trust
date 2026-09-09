# Insider Threat Detection with XGBoost & Semantic Reasoning

A hybrid approach combining **XGBoost** with **ontology-based reasoning** for explainable insider threat detection on the CERT 4.2 dataset.

## 🎯 Objective

Build an adaptive, explainable system that detects insider threats while providing context and reasoning behind each decision.

## 🧠 Key Features

- **ML Model**: XGBoost with SMOTE for class imbalance
- **Semantic Enrichment**: Converts raw features to risk indicators
- **Knowledge Graph**: RDF-based graph connecting users, events, risks
- **Explainability**: SHAP values for feature importance
- **Dynamic Policies**: Adaptive Zero Trust enforcement

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 91.8 |
| F1-Score | 0.910
| AUC-ROC | 0.978 |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ramshazameer016-ai/SemanticZeroTrust-Experiment.git
cd SemanticZeroTrust-Experiment