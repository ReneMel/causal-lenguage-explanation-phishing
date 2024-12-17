import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # Importar tqdm para la barra de progreso
import json
import os

class TinyLlamaPhishingClassifier:
    def __init__(self, model_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0', save_path="tinyllama_model.pt"):
        """
        Initialize TinyLlama for phishing classification.
        
        Args:
            model_path (str): Path to the TinyLlama model
            save_path (str): Path to save the model
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model (we'll use the causal LM and adapt it for classification)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Path to save the model
        self.save_path = save_path

    def preprocess_text(self, texts):
        """
        Preprocess texts for the model.
        
        Args:
            texts (list or str): Email text(s)
        
        Returns:
            dict: Tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )
    
    def classify_phishing(self, texts):
        """
        Classify if texts are phishing.
        
        Args:
            texts (list or str): Email text(s)
        
        Returns:
            tuple: (predictions, confidences)
        """
        inputs = self.preprocess_text(texts)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=1)
        
        return predictions.cpu().numpy(), confidences.cpu().numpy()
    
    def save_model(self):
        """
        Save the model state to the specified save_path.
        """
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

def process_phishing_dataset(input_path, output_path=None, save_model=False):
    """
    Process phishing email dataset with TinyLlama classification.
    
    Args:
        input_path (str): Path to input Parquet file
        output_path (str, optional): Path to save processed Parquet file
        save_model (bool): Whether to save the model after processing
    
    Returns:
        pd.DataFrame: Processed dataframe with TinyLlama classifications
    """
    df = pd.read_parquet(input_path)
    classifier = TinyLlamaPhishingClassifier()

    tinyllama_predictions = []
    tinyllama_confidences = []
    
    # Barra de progreso usando tqdm
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size), desc="Classifying Emails"):
        batch_texts = df['text'].iloc[i:i+batch_size].tolist()
        predictions, confidences = classifier.classify_phishing(batch_texts)
        tinyllama_predictions.extend(predictions)
        tinyllama_confidences.extend(confidences)
    
    df['tinyllama_predicted_label'] = tinyllama_predictions
    df['tinyllama_prediction_confidence'] = tinyllama_confidences
    df['prediction_match'] = (df['predicted_label'] == df['tinyllama_predicted_label'])
    df['explanation_summary'] = df.apply(generate_explanation_summary, axis=1)
    
    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"Processed file saved to {output_path}")
    
    if save_model:
        classifier.save_model()
    
    return df

def generate_explanation_summary(row):
    explanation = {
        'text_length': len(str(row['text'])),
        'original_predicted_label': bool(row['predicted_label']),
        'tinyllama_predicted_label': bool(row['tinyllama_predicted_label']),
        'true_label': bool(row['true_label']),
        'prediction_match': row['prediction_match'],
        'tinyllama_prediction_confidence': float(row['tinyllama_prediction_confidence']) if pd.notnull(row['tinyllama_prediction_confidence']) else None,
        'top_explanation_features': {}
    }
    explanation['shap_top_words'] = row['shap_top_words'] if isinstance(row['shap_top_words'], list) else []
    explanation['shap_word_importances'] = row['shap_word_importances'] if isinstance(row['shap_word_importances'], list) else []
    explanation['lime_top_words'] = row['lime_top_words'] if isinstance(row['lime_top_words'], list) else []
    explanation['lime_word_importances'] = row['lime_word_importances'] if isinstance(row['lime_word_importances'], list) else []
    return explanation

def analyze_dataset_predictions(df):
    explanations_df = pd.DataFrame(df['explanation_summary'].tolist())
    analysis = {
        'total_emails': len(df),
        'true_phishing_emails': explanations_df['true_label'].sum(),
        'original_predicted_phishing': explanations_df['original_predicted_label'].sum(),
        'tinyllama_predicted_phishing': explanations_df['tinyllama_predicted_label'].sum(),
        'prediction_consistency_rate': explanations_df['prediction_match'].mean(),
        'average_tinyllama_confidence': explanations_df['tinyllama_prediction_confidence'].mean()
    }
    return analysis

def main():
    input_file = './roberta-base_correct_predictions_with_explanations.parquet'
    output_file = 'tinyllama_processed_phishing.parquet'
    
    df = process_phishing_dataset(input_file, output_file, save_model=True)
    dataset_analysis = analyze_dataset_predictions(df)
    
    print("TinyLlama Prediction Analysis:")
    print(json.dumps(dataset_analysis, indent=2))
    print("\nExample Explanations:")
    for explanation in df['explanation_summary'][:5]:
        print(json.dumps(explanation, indent=2))

if __name__ == "__main__":
    main()
