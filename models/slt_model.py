#!/usr/bin/env python3
"""
Self-Learning Tutorial Model (SLT Model)
ClinicalBERT + RAG for Organ-Disease Classification
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import faiss
import torch.utils.data as data_utils

# Optimize transformers settings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class OrganDiseaseDataset(data_utils.Dataset):
    """Fast dataset with batch tokenization"""
    def __init__(self, dataframe, tokenizer, max_length=128):
        print(f"Creating dataset with {len(dataframe)} examples")
        self.data = dataframe.reset_index(drop=True)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.long)
        
        all_texts = self.data['input'].tolist()
        
        self.encodings = tokenizer(
            all_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    

class SLTModel:
    
    def __init__(self, model_name: str, num_labels: int, data_dir: str = "."):
        self.model_name = model_name
        self.num_labels = num_labels
        self.data_dir = Path(data_dir)
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.trainer = None
        self.embedding_model = None
        self.faiss_index = None
        self.disease_kb = None
        self.results = {}
        # Executed this on MAC M3, so device will be set to 'mps'
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_fp16 = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_fp16 = False
        else:
            self.device = torch.device("cpu")
            self.use_fp16 = False
            

    def load_data(self, train_file='train_organ_diseases.csv', val_file='val_organ_diseases.csv', sample_size=None):
        
        train_path = self.data_dir / train_file
        val_path = self.data_dir / val_file
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        
        # Sample data if requested
        if sample_size:
            self.train_df_clean = self.train_df.sample(n=min(sample_size, len(self.train_df)), random_state=42)
            self.val_df_clean = self.val_df.sample(n=min(sample_size//5, len(self.val_df)), random_state=42)
        else:
            self.train_df_clean = self.train_df.copy()
            self.val_df_clean = self.val_df.copy()
        
        return self

    def prepare_labels(self):
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Encode organs as labels
        self.label_encoder = LabelEncoder()
        all_organs = pd.concat([self.train_df_clean['organ'], self.val_df_clean['organ']])
        self.label_encoder.fit(all_organs)

        self.train_df_clean['label'] = self.label_encoder.transform(self.train_df_clean['organ'])
        self.val_df_clean['label'] = self.label_encoder.transform(self.val_df_clean['organ'])

        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        return self

    def create_datasets(self, max_length=128):
        
        self.train_dataset = OrganDiseaseDataset(self.train_df_clean, self.tokenizer, max_length)
        self.val_dataset = OrganDiseaseDataset(self.val_df_clean, self.tokenizer, max_length)

        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")

        return self

    def train(self, epochs=2, batch_size=8, output_dir='./results'):
        # Load model
        num_labels = len(self.label_encoder.classes_)
        
        start_load = time.time()
        # Use trust_remote_code=True to bypass torch.load security check (temporary workaround)
        # Better solution: upgrade torch to >=2.6.0
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            use_safetensors=True  # Prefer safetensors format if available
        )
        self.model.to(self.device)

        # Define metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            fp16=self.use_fp16,
            dataloader_num_workers=0,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            report_to="none",
        )

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )

        # Train

        try:
            train_result = self.trainer.train()
            self.results['train_metrics'] = train_result.metrics
            
            # Save model
            save_path = './clinicalbert_organ_classifier'
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model saved to: {save_path}")
            return self
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_model(self):
       
        eval_results = self.trainer.evaluate()
        self.results['eval_metrics'] = eval_results
        
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Get predictions
        predictions = self.trainer.predict(self.val_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = self.val_df_clean['label'].values
        
        self.results['predictions'] = predicted_labels
        self.results['true_labels'] = true_labels
        
        # Classification report
        
        # Get unique labels present in the validation set
        unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))
        target_names = [self.label_encoder.classes_[i] for i in unique_labels]
        
        report = classification_report(
            true_labels, 
            predicted_labels,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
        print(report)
        self.results['classification_report'] = report
        
        return self

    # RAG System for Disease Retrieval
    # Used L6 model for embeddings
    def setup_rag(self, diagnoses_file=None):
       
        # Load embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if self.device.type == 'cuda':
            self.embedding_model = self.embedding_model.to(self.device)
        
        # Prepare disease knowledge base from training data
        # Extract unique ICD codes and diseases from training data
        disease_data = []
        for _, row in self.train_df.iterrows():
            if 'icd_codes' in row and pd.notna(row['icd_codes']):
                # Parse ICD codes if they're strings
                try:
                    icd_codes = eval(row['icd_codes']) if isinstance(row['icd_codes'], str) else row['icd_codes']
                    for icd in icd_codes:
                        disease_data.append({'ICD9_CODE': icd, 'LONG_TITLE': row['output']})
                except:
                    disease_data.append({'ICD9_CODE': 'Unknown', 'LONG_TITLE': row['output']})
        
        self.disease_kb = pd.DataFrame(disease_data).drop_duplicates()
        print(f"Knowledge base created with {len(self.disease_kb)} disease entries")
        
        # Create embeddings
        disease_texts = self.disease_kb['LONG_TITLE'].tolist()
        disease_embeddings = self.embedding_model.encode(
            disease_texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = disease_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(disease_embeddings.astype('float32'))
        
        return self

    def retrieve_diseases_rag(self, query, organ, top_k=10):
        if self.embedding_model is None or self.faiss_index is None:
            raise ValueError("RAG system not initialized. Call setup_rag() first.")
        
        # Enhance query with organ context
        enhanced_query = f"Diseases affecting the {organ}: {query}"
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([enhanced_query], convert_to_numpy=True)[0]
        
        # Search in FAISS
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        
        # Get top diseases
        retrieved_diseases = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.disease_kb):
                disease_info = self.disease_kb.iloc[idx]
                retrieved_diseases.append({
                    'icd9_code': disease_info['ICD9_CODE'],
                    'disease_name': disease_info['LONG_TITLE'],
                    'similarity_score': float(1 / (1 + distance))
                })
        
        return retrieved_diseases

    def predict_organ_and_diseases(self, clinical_text, top_k_diseases=10):
        """End-to-end prediction: organ classification + disease retrieval"""
        # Predict organ
        inputs = self.tokenizer(
            clinical_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_label].item()
        
        predicted_organ = self.label_encoder.inverse_transform([predicted_label])[0]
        
        # Retrieve diseases using RAG
        diseases = self.retrieve_diseases_rag(clinical_text, predicted_organ, top_k=top_k_diseases)
        
        return {
            'predicted_organ': predicted_organ,
            'confidence': confidence,
            'diseases': diseases
        }

    def evaluate_end_to_end(self, sample_size=200):
        
        correct_predictions = 0
        total_predictions = 0
        sample_size = min(sample_size, len(self.val_df_clean))
        
        for idx, row in self.val_df_clean.head(sample_size).iterrows():
            text = row['input']
            true_organ = row['organ']
            
            result = self.predict_organ_and_diseases(text)
            predicted_organ = result['predicted_organ']
            
            if predicted_organ == true_organ:
                correct_predictions += 1
            total_predictions += 1
            
            if total_predictions % 50 == 0:
                print(f"Processed {total_predictions}/{sample_size} samples")
        
        accuracy = correct_predictions / total_predictions
        
        self.results['e2e_accuracy'] = accuracy
        self.results['e2e_correct'] = correct_predictions
        self.results['e2e_total'] = total_predictions
        
        print(f"End-to-End Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        return self

    def plot_results(self, save_dir='./plots'):
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Confusion Matrix
        if 'predictions' in self.results and 'true_labels' in self.results:
            fig, ax = plt.subplots(figsize=(14, 12))
            cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_, 
                       yticklabels=self.label_encoder.classes_,
                       cbar_kws={'label': 'Count'}, ax=ax)
            ax.set_title('Confusion Matrix - Organ Classification', fontsize=16, pad=20)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            confusion_path = save_path / 'confusion_matrix.png'
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {confusion_path}")
            plt.close()
        
        # 2. Accuracy Comparison Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = {
            'Training': self.results.get('train_metrics', {}).get('train_loss', 0),
            'Validation': self.results.get('eval_metrics', {}).get('eval_accuracy', 0),
            'End-to-End': self.results.get('e2e_accuracy', 0)
        }
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax.bar(metrics.keys(), [metrics['Validation'], metrics['End-to-End'], 
                                        self.results.get('eval_metrics', {}).get('eval_f1', 0)], 
                      color=colors[:3])
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics', fontsize=16, pad=20)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        accuracy_path = save_path / 'accuracy_comparison.png'
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved accuracy comparison: {accuracy_path}")
        plt.close()
        
        # 3. Per-Class Performance
        if 'predictions' in self.results and 'true_labels' in self.results:
            from sklearn.metrics import precision_recall_fscore_support
            
            precision, recall, f1, support = precision_recall_fscore_support(
                self.results['true_labels'], 
                self.results['predictions'],
                labels=range(len(self.label_encoder.classes_)),
                zero_division=0
            )
            
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(self.label_encoder.classes_))
            width = 0.25
            
            bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
            bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
            bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
            
            ax.set_xlabel('Organ Class', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            perclass_path = save_path / 'per_class_performance.png'
            plt.savefig(perclass_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Summary Metrics Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Training Summary', fontsize=18, fontweight='bold', y=0.995)
        
        # Accuracy
        eval_metrics = self.results.get('eval_metrics', {})
        ax1.bar(['Accuracy'], [eval_metrics.get('eval_accuracy', 0)], color='#3498db')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.set_title('Validation Accuracy')
        ax1.text(0, eval_metrics.get('eval_accuracy', 0) + 0.02, 
                f"{eval_metrics.get('eval_accuracy', 0):.4f}", 
                ha='center', fontweight='bold')
        
        # Precision/Recall/F1
        metrics_data = [
            eval_metrics.get('eval_precision', 0),
            eval_metrics.get('eval_recall', 0),
            eval_metrics.get('eval_f1', 0)
        ]
        bars = ax2.bar(['Precision', 'Recall', 'F1'], metrics_data, 
                       color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title('Weighted Metrics')
        for i, bar in enumerate(bars):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{metrics_data[i]:.4f}', ha='center', fontweight='bold')
        
        # Training time
        training_time = self.results.get('training_time', 0)
        ax3.bar(['Training Time'], [training_time / 60], color='#9b59b6')
        ax3.set_ylabel('Minutes')
        ax3.set_title('Training Duration')
        ax3.text(0, (training_time / 60) + 0.5, 
                f"{training_time / 60:.2f} min", 
                ha='center', fontweight='bold')
        
        # Dataset info
        dataset_info = [len(self.train_dataset), len(self.val_dataset)]
        bars = ax4.bar(['Train', 'Validation'], dataset_info, color=['#3498db', '#2ecc71'])
        ax4.set_ylabel('Samples')
        ax4.set_title('Dataset Size')
        for i, bar in enumerate(bars):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                    f'{dataset_info[i]:,}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        summary_path = save_path / 'training_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Saved training summary: {summary_path}")
        plt.close()
        
        print(f"\nAll plots saved to: {save_path}")
        
        return self

    def save_results(self, filename='training_results.json'):
        """Save results to JSON file"""
        output = {
            'model_name': self.model_name,
            'device': str(self.device),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'training_time_minutes': self.results.get('training_time', 0) / 60,
            'eval_metrics': self.results.get('eval_metrics', {}),
            'e2e_accuracy': self.results.get('e2e_accuracy', 0),
            'e2e_correct': self.results.get('e2e_correct', 0),
            'e2e_total': self.results.get('e2e_total', 0),
        }
        
        # Convert numpy types to native Python types
        for key in output['eval_metrics']:
            if isinstance(output['eval_metrics'][key], (np.floating, np.integer)):
                output['eval_metrics'][key] = float(output['eval_metrics'][key])
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filename}")

        return self

    def run_full_pipeline(self, sample_size=10000, epochs=2, batch_size=8):
        try:
            self.load_data(sample_size=sample_size)
            self.prepare_labels()
            self.create_datasets()
            self.train(epochs=epochs, batch_size=batch_size)
            self.evaluate_model()
            self.setup_rag()
            self.evaluate_end_to_end(sample_size=200)
            self.plot_results()
            self.save_results()
            print("\n Pipeline completed successfully.")
            
        except Exception as e:
            print(f"\n Pipeline failed: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    # Using smaller sample size and epochs for quick testing
    # Using ClinicalBERT model
    slt_model = SLTModel(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_labels=10,
        data_dir="."
    )
    
    # Run full pipeline
    slt_model.run_full_pipeline(
        sample_size=10000,  # Use 10000 samples for faster training
        epochs=2,           # Train for 2 epochs
        batch_size=8        # Batch size
    )
