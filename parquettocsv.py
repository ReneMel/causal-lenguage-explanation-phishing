import pandas as pd

# Nombre del archivo Parquet
parquet_file = "./roberta-base_correct_predictions_with_explanations.parquet"  # Cambia esto según sea necesario
csv_file = "roberta-base_correct_predictions_with_explanations.csv"          # Cambia esto también si quieres otro nombre

# Leer el archivo Parquet
df = pd.read_parquet(parquet_file)

# Guardar como CSV
df.to_csv(csv_file, index=False)
print(df.head())

print(f"Archivo convertido exitosamente de {parquet_file} a {csv_file}.")