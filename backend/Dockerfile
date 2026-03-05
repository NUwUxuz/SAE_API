FROM python:3.11-slim 
# Note : Privilégie 3.11-slim ou 3.12-slim pour des images plus légères et stables. 
# La version 3.14 est très récente/expérimentale.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie le contenu du dossier local "app" dans le dossier "/app" du conteneur
COPY . .

EXPOSE 8000

# Si ton fichier s'appelle main.py et est à la racine de ton dossier /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]