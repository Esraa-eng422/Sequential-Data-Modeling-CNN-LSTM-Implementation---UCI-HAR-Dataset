# CNN-LSTM for Human Activity Recognition (HAR)

## 📚 Project Overview

This project implements a **Hybrid CNN-LSTM Architecture** for Human Activity Recognition (HAR) using smartphone sensor data. The implementation follows the architecture described in **Lecture 4: Sequential Data Modeling** by Prof. Noha El-Attar.

---

## 🎯 Objectives

- Implement a CNN-LSTM hybrid model as per Lecture 4 (Slide 11-12)
- Classify 6 human activities using sensor data
- Demonstrate understanding of Spatial Feature Extraction (CNN) and Temporal Dependencies (LSTM)
- Achieve accuracy between 85% - 92% on UCI HAR Dataset

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CNN-LSTM Architecture                     │
│                     (Lecture 4, Slide 11)                    │
├─────────────────────────────────────────────────────────────┤
│  Input Layer → (Samples, Time Steps, Features)              │
│       ↓                                                      │
│  CNN Layer → Extracts Spatial Features (Conv1D + Pooling)   │
│       ↓                                                      │
│  LSTM Layer → Captures Temporal Dependencies (128 units)    │
│       ↓                                                      │
│  Dense Layer → Final Classification (6 activities)          │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Flow (Based on Lecture 4):

| Layer | Purpose | Lecture Reference |
|-------|---------|-------------------|
| **CNN** | Spatial Feature Extraction | Slide 11 |
| **LSTM** | Temporal Dependencies | Slide 4-5 |
| **Dense** | Final Output/Classification | Slide 11 |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | UCI Human Activity Recognition with Smartphones |
| **Source** | Kaggle / UCI ML Repository |
| **Samples** | 10,299 total (7,352 train + 2,947 test) |
| **Features** | 561 sensor features per sample |
| **Classes** | 6 activities |
| **Sensor Type** | Accelerometer & Gyroscope (3-axis) |

### Activity Classes:

| Label | Activity |
|-------|----------|
| 0 | Walking |
| 1 | Walking Upstairs |
| 2 | Walking Downstairs |
| 3 | Sitting |
| 4 | Standing |
| 5 | Laying |

---

## 🛠️ Requirements

```python
# Core Libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Deep Learning
tensorflow>=2.13.0
keras>=2.13.0
```

### Platform:
- **Kaggle Notebook** (Recommended - Free GPU: Tesla T4)
- **Google Colab** (Alternative)
- **Local Machine** (Requires GPU for faster training)

---

## 🚀 How to Run

### Option 1: Kaggle (Recommended)

1. **Create Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Create Notebook**: Click "Create" → "Notebook"
3. **Add Dataset**: 
   - Click "Add Data" on the right panel
   - Search: `Human Activity Recognition with Smartphones`
   - Click "+" to add
4. **Copy Code**: Copy all 7 cells from the implementation
5. **Run All**: Click "Run All" or press `Shift + Enter` for each cell

### Option 2: Google Colab

1. Visit [colab.research.google.com](https://colab.research.google.com)
2. Upload dataset to Google Drive
3. Mount Drive and load data
4. Run the notebook cells

### Option 3: Local Machine

```bash
# Clone repository (if applicable)
git clone <repository-url>
cd cnn-lstm-har

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook HAR_CNN_LSTM.ipynb
```

---

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| **Training Accuracy** | 88% - 95% |
| **Test Accuracy** | 85% - 92% |
| **Training Time** | 5-15 minutes (with GPU) |
| **Epochs** | 20-40 (with EarlyStopping) |

### Performance by Activity:

| Activity | Expected Accuracy | Difficulty |
|----------|------------------|------------|
| Sitting | 95%+ | Easy (static pattern) |
| Standing | 95%+ | Easy (static pattern) |
| Laying | 95%+ | Easy (static pattern) |
| Walking | 90%+ | Medium |
| Walking Upstairs | 85%+ | Hard (similar to downstairs) |
| Walking Downstairs | 85%+ | Hard (similar to upstairs) |

---

## 🔬 Theoretical Foundation (Lecture 4)

### 1. Why CNN-LSTM? (Slide 11)

| Component | Role | Benefit |
|-----------|------|---------|
| **CNN** | Extracts spatial features from sensor data | Detects local patterns in 561 features |
| **LSTM** | Captures temporal dependencies | Learns sequence patterns over time |

### 2. LSTM Advantages (Slide 4-5)

- ✅ **Overcomes Vanishing Gradient Problem**
- ✅ **Captures Long-term Dependencies**
- ✅ **Uses 3 Gates**: Forget, Input, Output

### 3. LSTM vs GRU (Slide 5)

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| Complexity | Higher | Lower |
| Performance | Better on complex sequences | Faster training |
| **Our Choice** | ✅ LSTM | - |

### 4. Why Not Transformers? (Slide 5)

- Transformers use **Self-Attention** for parallel processing
- For HAR task, CNN-LSTM is more suitable because:
  - Sensor data has clear **local patterns** (CNN excels)
  - Temporal dependencies are **moderate length** (LSTM handles well)
  - **Less computational resources** needed

---

## 📁 File Structure

```
cnn-lstm-har/
├── README.md                 # This file
├── HAR_CNN_LSTM.ipynb        # Main notebook (7 cells)
├── requirements.txt          # Python dependencies
├── models/
│   └── cnn_lstm_har_model.h5 # Saved model (after training)
└── results/
    ├── accuracy_plot.png     # Training/Validation Accuracy
    ├── loss_plot.png         # Training/Validation Loss
    └── confusion_matrix.png  # Classification Results
```

---

## 🧪 Code Structure (7 Cells)

| Cell | Purpose | Key Components |
|------|---------|----------------|
| 1 | Imports & Setup | TensorFlow, Keras, GPU check |
| 2 | Load Data | Kaggle path, CSV loading |
| 3 | Preprocessing | LabelEncoder, StandardScaler, Reshape |
| 4 | Build Model | CNN → LSTM → Dense (Slide 11) |
| 5 | Train Model | EarlyStopping, Adam optimizer |
| 6 | Evaluation | Accuracy, Confusion Matrix, Plots |
| 7 | Theory | Lecture 4 connection explanation |

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. ✅ **RNN Fundamentals** (Lecture 4, Slide 1-2)
2. ✅ **LSTM Architecture & Gates** (Slide 3-5)
3. ✅ **CNN-LSTM Hybrid Design** (Slide 11-12)
4. ✅ **Sequential Data Preprocessing**
5. ✅ **Model Evaluation & Visualization**
6. ✅ **Connection between Theory and Implementation**

---

## 📝 Key Formulas (From Lecture 4)

### LSTM Gates:

```
Forget Gate:     fₜ = σ(Wₓf·xₜ + Wₕf·hₜ₋₁ + bᶠ)
Input Gate:      iₜ = σ(Wₓi·xₜ + Wₕi·hₜ₋₁ + bⁱ)
Cell Candidate:  C̃ₜ = tanh(WₓC·xₜ + WₕC·hₜ₋₁ + bᶜ)
Cell State:      Cₜ = fₜ·Cₜ₋₁ + iₜ·C̃ₜ
Output Gate:     oₜ = σ(Wₓo·xₜ + Wₕo·hₜ₋₁ + bᵒ)
Hidden State:    hₜ = oₜ·tanh(Cₜ)
```

### GRU Comparison:

```
Reset Gate:      rₜ = σ(Wₓr·xₜ + Wₕr·hₜ₋₁ + bʳ)
Update Gate:     zₜ = σ(Wₓz·xₜ + Wₕz·hₜ₋₁ + bᶻ)
Candidate:       h̃ₜ = tanh(Wₕ·xₜ + Uₕ·(rₜ·hₜ₋₁) + bʰ)
Hidden State:    hₜ = (1-zₜ)·hₜ₋₁ + zₜ·h̃ₜ
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| FileNotFoundError | Check Kaggle input path in Cell 2 |
| TypeError (str - int) | Use LabelEncoder in Cell 3 (fixed version) |
| ValueError (ndim=3) | Remove Flatten layer in Cell 4 (fixed version) |
| Low Accuracy | Increase epochs or adjust filters |
| Memory Error | Reduce batch_size to 32 |
| GPU Not Detected | Check TensorFlow version, restart runtime |

---

## 📚 References
