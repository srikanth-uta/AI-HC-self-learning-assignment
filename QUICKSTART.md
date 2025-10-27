# Quick Start Guide - SLT Model Training

## Overview
This project trains a ClinicalBERT model with RAG (Retrieval-Augmented Generation) for organ-disease classification using MIMIC-III data.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data Files
Make sure you have these files in the project directory:
- `train_organ_diseases.csv`
- `val_organ_diseases.csv`

## Running the Model

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd models
python slt_model.py
```

This will:
1. ✅ Load training and validation data
2. ✅ Prepare labels and tokenize data
3. ✅ Train ClinicalBERT model
4. ✅ Evaluate model performance
5. ✅ Setup RAG system for disease retrieval
6. ✅ Run end-to-end evaluation
7. ✅ Generate visualization charts
8. ✅ Save results

### Option 2: Custom Usage in Python

```python
from models.slt_model import SLTModel

# Initialize model
model = SLTModel(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    num_labels=10,
    data_dir="."
)

# Run full pipeline
model.run_full_pipeline(
    sample_size=10000,  # Number of training samples
    epochs=2,           # Training epochs
    batch_size=8        # Batch size
)
```

### Option 3: Step-by-Step Execution

```python
from models.slt_model import SLTModel

# Initialize
model = SLTModel(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    num_labels=10
)

# Run each step individually
model.load_data(sample_size=10000)
model.prepare_labels()
model.create_datasets()
model.train(epochs=2, batch_size=8)
model.evaluate_model()
model.setup_rag()
model.evaluate_end_to_end(sample_size=200)
model.plot_results()
model.save_results()
```

## Outputs

After training, you'll find:

### 📁 Models
- `./clinicalbert_organ_classifier/` - Trained model weights and tokenizer

### 📊 Visualizations (in `./plots/`)
1. `confusion_matrix.png` - Confusion matrix for organ classification
2. `accuracy_comparison.png` - Comparison of different accuracy metrics
3. `per_class_performance.png` - Precision, Recall, F1 per organ class
4. `training_summary.png` - Overall training summary dashboard

### 📄 Results
- `training_results.json` - Complete metrics and configuration

### 📝 Logs
- `./logs/` - TensorBoard training logs
- `./results/` - Model checkpoints

## Making Predictions

```python
# Load trained model
model = SLTModel(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    num_labels=10
)

# Load from saved checkpoint
model.model = AutoModelForSequenceClassification.from_pretrained(
    './clinicalbert_organ_classifier'
)
model.tokenizer = AutoTokenizer.from_pretrained(
    './clinicalbert_organ_classifier'
)

# Predict
result = model.predict_organ_and_diseases(
    "Patient presents with chest pain and shortness of breath."
)

print(f"Predicted Organ: {result['predicted_organ']}")
print(f"Confidence: {result['confidence']:.4f}")
print("\nRelevant Diseases:")
for disease in result['diseases'][:5]:
    print(f"  - {disease['disease_name']} (ICD9: {disease['icd9_code']})")
```

## Performance Optimization

### For Faster Training
- Reduce `sample_size` (default: 10000)
- Reduce `epochs` (default: 2)
- Reduce `batch_size` (default: 8)

### For Better Accuracy
- Increase `sample_size` to use full dataset
- Increase `epochs` (try 3-5)
- Increase `batch_size` if you have more GPU memory

### Device Selection
The model automatically detects and uses:
- **CUDA** (NVIDIA GPU) - Fastest
- **MPS** (Apple Silicon M1/M2/M3) - Fast
- **CPU** - Slower but works everywhere

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
model.run_full_pipeline(batch_size=4)  # Instead of 8
```

### Import Errors
```bash
pip install --upgrade transformers torch sentence-transformers faiss-cpu
```

### Slow Training
- First run downloads ~400MB model (one-time)
- Subsequent runs load from cache (fast)
- Training 10k samples takes ~5-15 minutes on GPU

## Expected Results

With default settings (10k samples, 2 epochs):
- **Validation Accuracy**: ~0.75-0.85
- **End-to-End Accuracy**: ~0.70-0.80
- **F1 Score**: ~0.72-0.82
- **Training Time**: ~5-15 minutes (GPU) or ~30-60 minutes (CPU)

## Next Steps

1. **Fine-tune hyperparameters** - Adjust epochs, batch size, learning rate
2. **Use full dataset** - Set `sample_size=None` for maximum accuracy
3. **Experiment with models** - Try different BERT variants
4. **Improve RAG** - Add more disease knowledge sources
5. **Deploy model** - Create API endpoint for predictions

## Support

For issues or questions, check:
- `MIMIC_LLM_setup.ipynb` - Detailed setup notebook
- `MIMIC_LLM_MODEL.ipynb` - Model training notebook
- `README.md` - Project overview
