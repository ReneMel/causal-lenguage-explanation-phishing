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
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración para control de explicaciones
generate_explanations = True  # Cambiar a True para generar SHAP y LIME explanations

# Cargar el dataset
dataset = load_dataset("renemel/compiled-phishing-dataset", split="train")
df = dataset.to_pandas()

# Codificar las etiquetas con LabelEncoder
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Dividir en conjunto de entrenamiento y prueba
train_size = 0.7
train_df, test_df, train_labels, test_labels = train_test_split(
    df['text'], df['type'], test_size=train_size, random_state=42
)

# Clase para el Dataset de Emails
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
        if not text:  # Validar texto vacío
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

# Función para predicción de probabilidades
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

# Función para agregar explicaciones con SHAP y LIME con manejo de errores
def add_explanations(df, model, tokenizer, device):
    if not generate_explanations:  # Salir si no se deben generar explicaciones
        return df

    def shap_predict_wrapper(texts):
        return predict_proba(texts, model, tokenizer, device)
    
    try:
        masker = shap.maskers.Text(tokenizer)
        explainer_shap = shap.Explainer(shap_predict_wrapper, masker)
        explainer_lime = LimeTextExplainer(class_names=['Negative', 'Positive'])
    except Exception as e:
        logger.error(f"Error initializing SHAP/LIME explainers: {e}")
        return df

    shap_values_list = []
    lime_values_list = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating SHAP and LIME explanations"):
        text = row['text'].strip()
        if not text:
            text = "[PAD]"

        try:
            # SHAP Explanation
            shap_values = explainer_shap([text])
            shap_values_list.append(shap_values.values[0].tolist())
        except Exception as e:
            logger.error(f"SHAP explanation error for index {index}: {e}")
            shap_values_list.append(None)

        try:
            # LIME Explanation
            lime_exp = explainer_lime.explain_instance(
                text,
                lambda x: predict_proba(x, model, tokenizer, device),
                num_features=10
            )
            lime_values = {word: weight for word, weight in lime_exp.as_list()}
            lime_values_list.append(lime_values)
        except Exception as e:
            logger.error(f"LIME explanation error for index {index}: {e}")
            lime_values_list.append(None)

    df['shap_values'] = shap_values_list
    df['lime_values'] = lime_values_list
    return df

# Modelo y tokenizador
model_names = ['roberta-base']

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

    # Entrenamiento
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

    # Evaluación
    model.eval()
    false_negatives = []
    true_negatives = []
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
                elif pred == 0 and label == 0:  
                    true_negatives.append({'text': test_df.iloc[global_index], 'true_label': label, 'predicted_label': pred})
                elif pred == label:  
                    correct_predictions.append({'text': test_df.iloc[global_index], 'true_label': label, 'predicted_label': pred})

                all_labels.append(label)
                all_preds.append(pred)

    # Guardar resultados con explicaciones si es necesario
    tn_fn_df = pd.DataFrame(false_negatives + true_negatives)
    tp_tn_df = pd.DataFrame(correct_predictions + true_negatives)

    try:
        tn_fn_df = add_explanations(tn_fn_df, model, tokenizer, device)
        tp_tn_df = add_explanations(tp_tn_df, model, tokenizer, device)

        tn_fn_df.to_parquet(f'{model_name}_tn_fn.parquet', index=False)
        tp_tn_df.to_parquet(f'{model_name}_tp_tn.parquet', index=False)
        print(f"Parquet files saved for {model_name}.")
    except Exception as e:
        logger.error(f"Error saving explanation results for {model_name}: {e}")