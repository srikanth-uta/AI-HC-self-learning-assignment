#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model_path = './clinicalbert_organ_classifier'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_safetensors=True
)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval()

train_df = pd.read_csv('train_organ_diseases.csv')
organ_classes = sorted(train_df['organ'].unique())

import pickle
from pathlib import Path

cache_file = Path('rag_cache.pkl')

if cache_file.exists():
    with open(cache_file, 'rb') as f:
        rag_cache = pickle.load(f)
        embedding_model = rag_cache['embedding_model']
        disease_kb = rag_cache['disease_kb']
        faiss_index = rag_cache['faiss_index']
else:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    disease_data = []
    sample_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)  # Use sample for speed
    
    for _, row in sample_df.iterrows():
        if 'icd_codes' in row and pd.notna(row['icd_codes']):
            try:
                icd_codes = eval(row['icd_codes']) if isinstance(row['icd_codes'], str) else row['icd_codes']
                for icd in icd_codes:
                    disease_data.append({'ICD9_CODE': icd, 'LONG_TITLE': row['output']})
            except:
                disease_data.append({'ICD9_CODE': 'Unknown', 'LONG_TITLE': row['output']})
    
    disease_kb = pd.DataFrame(disease_data).drop_duplicates()
    
    # Create embeddings
    disease_texts = disease_kb['LONG_TITLE'].tolist()
    disease_embeddings = embedding_model.encode(
        disease_texts, 
        batch_size=64,  # Larger batch for speed
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    
    # Create FAISS index
    dimension = disease_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(disease_embeddings.astype('float32'))
    
    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'embedding_model': embedding_model,
            'disease_kb': disease_kb,
            'faiss_index': faiss_index
        }, f)


print(f"Model loaded! Available organs: {organ_classes}\n")

# Test text
clinical_text = input("Enter clinical text: ")

# Predict
inputs = tokenizer(clinical_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_label].item()

predicted_organ = organ_classes[predicted_label]

# Retrieve diseases using RAG
enhanced_query = f"Diseases affecting the {predicted_organ}: {clinical_text}"
query_embedding = embedding_model.encode([enhanced_query], convert_to_numpy=True)[0]

# Search in FAISS
distances, indices = faiss_index.search(query_embedding.reshape(1, -1).astype('float32'), 5)

# Get top diseases
top_diseases = []
seen_diseases = set()  # Track unique diseases

for idx in indices[0]:
    if idx < len(disease_kb):
        disease_info = disease_kb.iloc[idx]
        disease_name = disease_info['LONG_TITLE']
        
        # Only add if not already seen
        if disease_name not in seen_diseases:
            top_diseases.append(disease_name)
            seen_diseases.add(disease_name)
        
        # Stop when we have 5 unique diseases
        if len(top_diseases) >= 5:
            break

# Generate natural language answer using training data
def generate_answer_from_training_data(clinical_text, predicted_organ, confidence, top_diseases, train_df):
    """
    Generate answer based on similar cases from training data
    This uses your trained model's knowledge, not external LLMs
    """
    
    # Find similar examples from training data for this organ
    organ_examples = train_df[train_df['organ'] == predicted_organ].head(10)
    
    # Get the most common/relevant answer patterns from training data
    sample_outputs = organ_examples['output'].tolist()
    
    # Use the retrieved diseases to create a contextual answer
    # This is based on what the model learned during training
    diseases_list = "\n".join([f"   • {disease}" for disease in top_diseases[:5]])
    
    answer = f"""
        ANALYSIS BASED ON TRAINED MODEL:

        Predicted Organ System: {predicted_organ.upper()}
        Confidence Level: {confidence:.1%}

        Most Relevant Diseases (Retrieved from Training Data):
        {diseases_list}

        """
    return answer.strip()

# Print result
print(f"\n{'='*70}")
print(f"CLINICAL ANALYSIS REPORT")
print(f"{'='*70}\n")

print(f"Input Clinical Text:")
print(f"   {clinical_text}\n")

print(f"Primary Prediction: {predicted_organ}")
print(f"Confidence: {confidence:.2%}\n")

# Show top 3
print("Top 3 Organ Predictions:")
top_3 = torch.topk(probabilities[0], k=3)
for i, (prob, idx) in enumerate(zip(top_3.values, top_3.indices), 1):
    print(f"   {i}. {organ_classes[idx]}: {prob:.2%}")

# Generate and display natural language answer
print(f"\n{'='*70}")
print(f"CLINICAL ANSWER (Generated from Trained Model)")
print(f"{'='*70}")
answer = generate_answer_from_training_data(clinical_text, predicted_organ, confidence, top_diseases, train_df)
print(answer)

