# Hayden-Chu-MUSA-6950-AI-for-Urban-Sustainability-Final-Project-
# 🐾 Wildlife Image Classification for Conservation Monitoring

This project explores the use of deep learning and traditional machine learning models for automated wildlife species classification from images. The study is motivated by real-world ecological challenges—such as monitoring elusive or rare species with limited training data—and demonstrates how tools like transfer learning, CNNs, and feature-based classification can support biodiversity monitoring at scale.

## 🧠 Project Objectives

- Classify wildlife species (monkeys and birds) from images using various machine learning methods.
- Compare the performance of:
  - Custom Convolutional Neural Networks (CNNs)
  - Transfer Learning with Xception and MobileNetV2
  - Traditional ML models (Random Forest, K-Nearest Neighbors)
  - Basic fully connected neural networks (MLP)
- Evaluate models under both **data-scarce** (monkey dataset) and **moderately sized** (bird dataset) scenarios.
- Provide insights into model generalization, overfitting, and species-level misclassifications.

## 📂 Contents

- `image_classification_python_final.ipynb` – Main Jupyter notebook for model training, evaluation, and visualization.
- `data/` – Directory for training, validation, and test datasets (not included in repo due to size). Links to data are provided in Jupyter notebook
- `assets/` – Model architecture diagrams, performance graphs, and prediction visualizations.
- `report.docx` – Final paper summarizing methodology, results, and discussion.
- `presentation.pptx` – Project summary slides (4–5 minute oral presentation).
- `README.md` – This file.

## 🔍 Datasets

1. **Monkey Dataset**
   - Extremely limited training data (~10 images per species).
   - Used to simulate species that are rare or elusive.
   - 10 species total.

2. **Bird Dataset**
   - 500 species with ~100 images each.
   - Allows testing under more realistic and diverse ecological conditions.

## 🛠️ Methods

- **Image Augmentation**: Random rotations, flips, zooming, and brightness adjustments to expand training sets.
- **CNNs**: Built from scratch for baseline benchmarking.
- **Transfer Learning**:
  - `Xception` for monkeys
  - `MobileNetV2` for birds
- **Traditional ML**:
  - Trained on CNN-extracted features.
  - Models: Random Forest, K-Nearest Neighbors, Fully Connected Neural Network (MLP).

## 🔄 Step-by-Step Workflow

### 1. 🧼 Data Preparation
- Organize images into `train`, `val`, and `test` directories.
- Each subdirectory should contain species folders (e.g., `n0`, `n1`, ..., `n9`).
- Use `ImageDataGenerator` in Keras for loading and real-time image augmentation.

### 2. 🧪 Data Augmentation
To increase data diversity and prevent overfitting:
- Rotation
- Zooming
- Horizontal/vertical flipping
- Brightness/contrast adjustments

### 3. 🧠 Model Training

#### Option A: Custom CNN
- Built from scratch using Keras `Sequential`.
- Consists of convolutional, pooling, dropout, and dense layers.
- Trained for up to 200 epochs with early stopping.

#### Option B: Transfer Learning (Xception)
- Use pre-trained `Xception` from `keras.applications`.
- Freeze convolutional base, add new dense classification head.
- Trained for 15–40 epochs depending on dataset size.

#### Option C: Traditional ML Models
- Use Xception to extract deep features (via `.predict()`).
- Train classifiers on flattened feature vectors:
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Fully Connected Neural Network (MLP)

### 4. 📈 Evaluation
- Evaluate with accuracy, precision, recall, F1-score, sensitivity, and specificity.
- Display classification reports, confusion matrices, and sample predictions.
- Use Matplotlib and Seaborn for plotting.

### 5. 🔍 Visualizations
- Accuracy and loss curves for training/validation
- Confusion matrices and classification heatmaps
- Activation maps (CNN feature visualizations)
- Prediction overlays (True label vs. Predicted)

## 📊 Results Summary

| Model                  | Monkey Data | Bird Data |
|------------------------|-------------|-----------|
| CNN (Custom)           | 11.76%      | —         |
| Transfer Learning      | 10.29%      | 84.11%    |
| Random Forest          | 35%         | —         |
| K-Nearest Neighbors    | 21%         | —         |
| Fully Connected Network| **73%**     | —         |

---

## 🌿 Key Takeaways

- Transfer learning is essential in data-limited or high-diversity settings.
- Simpler models can outperform CNNs when sample size is extremely small.
- Feature-rich CNN embeddings can be repurposed for traditional ML models.
- Species-level misclassifications reveal where models need better augmentation or domain-specific fine-tuning.
