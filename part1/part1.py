import pandas as pd
import torch
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_curve, auc
from torch.optim import AdamW
import matplotlib.pyplot as plt
from datasets import load_dataset  

# Load the dataset
dataset = load_dataset("renemel/compiled-phishing-dataset", split="train")  
df = dataset.to_pandas()

# Encode labels
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Split into training and test sets
train_size = 0.7
train_df, test_df, train_labels, test_labels = train_test_split(
    df['text'], df['type'], test_size=train_size, random_state=42
)

# Email Dataset Class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]).strip()  # Remove whitespace
        if not text:  # Replace empty text
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

# Probability Prediction Function
def predict_proba(texts, model, tokenizer, device):
    probabilities = []
    for text in texts:
        text = text.strip()  # Validate empty text or spaces
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

# Explanation Addition Function
def add_explanations(df, model, tokenizer, device):
    def shap_predict_wrapper(texts):
        return predict_proba(texts, model, tokenizer, device)
    
    masker = shap.maskers.Text(tokenizer)
    explainer_shap = shap.Explainer(shap_predict_wrapper, masker)
    explainer_lime = LimeTextExplainer(class_names=['Negative', 'Positive'])

    shap_values_list = []
    lime_values_list = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating SHAP and LIME explanations"):
        text = row['text'].strip()  # Remove spaces
        
        # Improve handling of short texts
        if not text or len(text.split()) <= 2:
            shap_values_list.append({})
            lime_values_list.append({})
            continue

        try:
            # More robust method for LIME
            lime_exp = explainer_lime.explain_instance(
                text,
                lambda x: predict_proba(x, model, tokenizer, device),
                num_features=min(10, len(text.split())),  # Adjust number of features
                labels=(0, 1)  # Explicitly specify labels
            )
            lime_values = {word: weight for word, weight in lime_exp.as_list()}
            lime_values_list.append(lime_values)

            # SHAP
            shap_values = explainer_shap([text])
            shap_values_list.append(shap_values.values[0].tolist())

        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Exception: {e}")
            shap_values_list.append({})
            lime_values_list.append({})

    df['shap_values'] = shap_values_list
    df['lime_values'] = lime_values_list
    return df

# Model and tokenizer
model_names = ['roberta-base']
results = []

for model_name in model_names:
    print(f"Training and evaluating model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = EmailDataset(train_df, train_labels, tokenizer)
    test_dataset = EmailDataset(test_df, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

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

            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    # Evaluation
    model.eval()
    false_negatives = []
    false_positives = []  
    correct_predictions = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating Model')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.argmax(outputs.logits, dim=1)
            
            for idx, (pred, label) in enumerate(zip(predictions.cpu().numpy(), labels.cpu().numpy())):
                global_index = batch_idx * test_loader.batch_size + idx  
                if pred == 0 and label == 1:  
                    false_negatives.append({'text': test_df.iloc[global_index], 'true_label': label, 'predicted_label': pred})
                elif pred == 1 and label == 0:  
                    false_positives.append({'text': test_df.iloc[global_index], 'true_label': label, 'predicted_label': pred})
                elif pred == label:  
                    correct_predictions.append({'text': test_df.iloc[global_index], 'true_label': label, 'predicted_label': pred})

                all_labels.append(label)
                all_preds.append(pred)

    # Save results with explanations
    fn_df = pd.DataFrame(false_negatives)
    fp_df = pd.DataFrame(false_positives)
    cp_df = pd.DataFrame(correct_predictions)

    fn_df_with_explanations = add_explanations(fn_df, model, tokenizer, device)
    fp_df_with_explanations = add_explanations(fp_df, model, tokenizer, device)
    cp_df_with_explanations = add_explanations(cp_df, model, tokenizer, device)

    fn_df_with_explanations.to_parquet(f'{model_name}_falsenegatives_with_explanations.parquet', index=False)
    fp_df_with_explanations.to_parquet(f'{model_name}_falsepositives_with_explanations.parquet', index=False)
    cp_df_with_explanations.to_parquet(f'{model_name}_correctpredictions_with_explanations.parquet', index=False)

    print(f"False Negatives, False Positives, and Correct Predictions saved for {model_name}")