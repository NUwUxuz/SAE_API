FROM node:20-slim

WORKDIR /app

# On copie d'abord les fichiers de dépendances pour optimiser le cache Docker
COPY package*.json ./
RUN npm install

# On copie le reste du code
COPY . .

# On expose le port par défaut de Vite
EXPOSE 5173

# IMPORTANT : On force l'hôte à 0.0.0.0 pour que Docker laisse passer le trafic
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

