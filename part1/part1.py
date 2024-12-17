import pandas as pd
import torch
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm  
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from torch.optim import AdamW
from datasets import load_dataset
import logging
import json

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
dataset = load_dataset("renemel/compiled-phishing-dataset", split="train")
df = dataset.to_pandas()

# Preprocessing
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Split data
train_size = 0.7
train_df, test_df, train_labels, test_labels = train_test_split(
    df['text'], df['type'], test_size=train_size, random_state=42
)

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]).strip()
        if not text:  
            text = "[PAD]"
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def predict_proba(texts, model, tokenizer, device):
    probabilities = []
    for text in texts:
        text = text.strip()
        if not text:
            text = "[PAD]"
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=100,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probabilities.append(probs[0])

    return np.array(probabilities)

def generate_explanations(df, model, tokenizer, device):
    def shap_predict_wrapper(texts):
        return predict_proba(texts, model, tokenizer, device)

    try:
        masker = shap.maskers.Text(tokenizer)
        explainer_shap = shap.Explainer(shap_predict_wrapper, masker)
        explainer_lime = LimeTextExplainer(class_names=['Negative', 'Positive'])
    except Exception as e:
        logger.error(f"Error initializing explainers: {e}")
        return df

    shap_top_words, shap_word_importances = [], []
    lime_top_words, lime_word_importances = [], []

    for index in tqdm(range(df.shape[0]), desc="Generating explanations"):
        try:
            text = df.iloc[index]['text'].strip() or "[PAD]"

            # SHAP Explanations
            try:
                shap_values = explainer_shap([text])
                
                word_importances = np.abs(shap_values.values[0].flatten())
                tokens = tokenizer.encode(text, add_special_tokens=True)
                decoded_tokens = tokenizer.convert_ids_to_tokens(tokens)
                
                word_importance_dict = dict(zip(decoded_tokens, word_importances))
                sorted_words = sorted(word_importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                top_10_words = sorted_words[:10]
                
                shap_top_words.append(json.dumps([word for word, _ in top_10_words]))
                shap_word_importances.append(json.dumps([float(importance) for _, importance in top_10_words]))
                
            except Exception as shap_error:
                logger.error(f"SHAP explanation error for index {index}: {shap_error}")
                shap_top_words.append(json.dumps([]))
                shap_word_importances.append(json.dumps([]))

            # LIME Explanations
            try:
                lime_exp = explainer_lime.explain_instance(
                    text,
                    lambda x: predict_proba(x, model, tokenizer, device),
                    num_features=10
                )
                
                words = [word for word, _ in lime_exp.as_list()]
                importances = [abs(float(weight)) for _, weight in lime_exp.as_list()]
                
                lime_top_words.append(json.dumps(words))
                lime_word_importances.append(json.dumps(importances))
                
            except Exception as lime_error:
                logger.error(f"LIME explanation error for index {index}: {lime_error}")
                lime_top_words.append(json.dumps([]))
                lime_word_importances.append(json.dumps([]))

        except Exception as general_error:
            logger.error(f"General explanation error for index {index}: {general_error}")
            shap_top_words.append(json.dumps([]))
            shap_word_importances.append(json.dumps([]))
            lime_top_words.append(json.dumps([]))
            lime_word_importances.append(json.dumps([]))

    df['shap_top_words'] = shap_top_words
    df['shap_word_importances'] = shap_word_importances
    df['lime_top_words'] = lime_top_words
    df['lime_word_importances'] = lime_word_importances

    return df

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating Model'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Classification report and confusion matrix
    class_report = classification_report(all_labels, all_preds, target_names=le.classes_)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'true_labels': all_labels
    }

# Main training and evaluation script
def main():
    model_name = 'roberta-base'
    
    # Tokenizer and model preparation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(le.classes_)
    )

    # Prepare datasets
    train_dataset = EmailDataset(train_df, train_labels, tokenizer)
    test_dataset = EmailDataset(test_df, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    evaluation_results = evaluate_model(model, test_loader, device)

    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])

    # Create a results DataFrame
    results_df = pd.DataFrame({
        'text': test_df.reset_index(drop=True),
        'true_label': evaluation_results['true_labels'],
        'predicted_label': evaluation_results['predictions'],
        'shap_top_words': '',  # Inicializada con cadenas vacías
        'shap_word_importances': '',
        'lime_top_words': '',
        'lime_word_importances': ''
    })

    print(results_df.head())

    # Generate explanations for a sample of correct predictions
    correct_predictions = results_df[results_df['true_label'] == results_df['predicted_label']]
    
    # Sample 1000 correct predictions
    sample_correct = correct_predictions.sample(n=min(2000, len(correct_predictions)), random_state=42)
    
    # Generate explanations
    sample_with_explanations = generate_explanations(sample_correct, model, tokenizer, device)

    # Save results
    sample_with_explanations.to_parquet(f'{model_name}_correct_predictions_with_explanations.parquet', index=False)
    
    print(f"\nSaved explanations for {len(sample_with_explanations)} correct predictions")

if __name__ == "__main__":
    main()