import kagglehub

print("Iniciando descarga del dataset de Yelp...")
path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
print(f"\n¡Descarga completada!\nLos archivos están en: {path}")