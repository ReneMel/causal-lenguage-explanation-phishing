
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

hf_token="token"
login(token=hf_token)

def generate_llama_explanation(row, model, tokenizer, device):
    text = row['text']  
    shap_values = row['shap_values'] 
    lime_values = row['lime_values']  


    shap_highlights = [
        f'{word} ({round(abs(value[0]), 2)})' for word, value in zip(text.split(), shap_values) if abs(value[0]) > 0.2
    ]
    shap_str = ', '.join(shap_highlights) if shap_highlights else "No significant SHAP values."

  
    lime_highlights = [
        f'{word} ({round(value, 2)})' for word, value in lime_values.items() if value and abs(value) > 0.2
    ]
    lime_str = ', '.join(lime_highlights) if lime_highlights else "No significant LIME values."

    
    prompt = (
        f"The following email was classified as {'phishing' if row['true_label'] == 1 else 'not phishing'}.\n\n"
        f"Email content:\n{text}\n\n"
        f"SHAP highlights the following keywords and their weights: {shap_str}.\n\n"
        f"LIME highlights the following keywords and their weights: {lime_str}.\n\n"
        "Based on this information, explain why this email is classified as phishing or not phishing in detail:"
    )

    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=90, num_return_sequences=1, temperature=0.7)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return explanation

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    
    parquet_file = '../part1/roberta-base_correctpredictions_with_explanations.parquet'  
    df = pd.read_parquet(parquet_file)

    
    df_subset = df.head(100)

    
    explanations = []
    for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc="Generating Explanations"):
        explanation = generate_llama_explanation(row, model, tokenizer, device)
        explanations.append({
            'text': row['text'],
            'true_label': row['true_label'],
            'explanation': explanation
        })

    
    for explanation in explanations:
        print(f"Email Text: {explanation['text']}\n")
        print(f"Explanation: {explanation['explanation']}\n")
        print(f"Classification: {'Phishing' if explanation['true_label'] == 1 else 'Not Phishing'}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
