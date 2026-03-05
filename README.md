# 🎵 SAE - API de Recommandation Musicale (Fullstack)

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

## 📝 1. Description du Projet

Ce projet a été réalisé dans le cadre de la **SAE [Numéro/Nom]** à l'[Nom de ton école]. Il s'agit d'une application web complète (Fullstack) permettant de gérer une vaste bibliothèque musicale, de créer des playlists, et de générer des recommandations personnalisées basées sur l'historique d'écoute des utilisateurs.

**Pourquoi cette architecture ?** Le projet est conçu autour d'une architecture conteneurisée (Docker). Cela permet de séparer proprement les responsabilités (Base de données, API, Interface Web, Script d'importation) et de garantir que le code fonctionnera de la même manière sur n'importe quel ordinateur, sans avoir à installer manuellement Python, Node.js ou PostgreSQL.

**Fonctionnalités principales :**
* 🎧 **Catalogue Musical :** Consultation des pistes, albums et artistes.
* ⭐ **Espace Utilisateur :** Gestion des favoris, historiques d'écoutes et création de playlists.
* 🤖 **Recommandations :** Moteur suggérant de nouvelles pistes basé sur l'activité de l'utilisateur.
* 📊 **Optimisation :** Utilisation de vues matérialisées et de triggers SQL pour garantir des temps de réponse rapides de l'API.

---

## 🏗️ 2. Architecture Technique

Notre application s'appuie sur des technologies modernes et robustes :

* **Base de données (PostgreSQL 15) :** Choisi pour sa puissance sur les requêtes relationnelles complexes. Nous utilisons des Triggers et des Vues Matérialisées pour pré-calculer les statistiques d'écoute.
* **ETL / Peuplement (Python & Pandas) :** Un script automatisé nettoie les fichiers de données brutes (`.csv`) et remplit la base de données au premier lancement.
* **Backend (FastAPI & SQLAlchemy) :** Un framework Python ultra-rapide qui génère automatiquement sa propre documentation interactive (Swagger). SQLAlchemy sert d'ORM pour faire le pont entre Python et les requêtes SQL.
* **Frontend (React / Vite) :** Une interface utilisateur réactive et basée sur des composants.
* **Orchestration (Docker Compose) :** L'outil qui relie tous ces éléments entre eux de manière transparente.

---

## 📁 3. Structure du Dépôt

Voici comment s'organise le code source du projet :

```text
SAE_API/
├── docker-compose.yml      # 🎼 Le chef d'orchestre qui configure et relie tous les conteneurs.
│
├── bdd/                    # 🗄️ TOUT CE QUI CONCERNE LA BASE DE DONNÉES
│   ├── init.sql            # Script SQL exécuté à la création de la base (Tables, Vues, Triggers).
│   ├── data/               # ⚠️ Dossier pour vos CSV bruts (Ignorés par Git pour éviter de saturer le dépôt).
│   └── app/                # Scripts Python (cleaning.py, main.py...) pour injecter les CSV dans PostgreSQL.
│
├── backend/                # ⚙️ TOUT CE QUI CONCERNE L'API REST
│   ├── main.py             # Point d'entrée de FastAPI (Routes et configuration).
│   ├── models.py           # Définition des tables (ORM SQLAlchemy).
│   ├── schema.py           # Validation des données entrantes/sortantes (Pydantic).
│   ├── requirements.txt    # Liste des librairies Python requises.
│   └── Dockerfile          # Recette pour construire le conteneur Backend.
│
└── frontend/               # 🖥️ TOUT CE QUI CONCERNE L'INTERFACE UTILISATEUR
    ├── package.json        # Configuration Node.js et dépendances React.
    ├── src/                # Code source de l'application Web.
    └── Dockerfile          # Recette pour construire le conteneur Frontend.
```

---

## 🚀 4. Installation et Lancement

Cette section vous guide pas à pas pour installer l'environnement de développement sur votre machine locale. 

### 0. Prérequis
Avant de commencer, assurez-vous d'avoir installé sur votre machine :
- **Git** : Pour récupérer le code source.
- **Docker** et **Docker Compose** (via Docker Desktop) : Pour faire tourner les serveurs de manière isolée.

### 1. Récupérer le projet
Ouvrez votre terminal et exécutez les commandes suivantes :
```bash
git clone [https://github.com/](https://github.com/)[votre-nom-dutilisateur]/SAE_API.git
cd SAE_API
```
> **Pourquoi ?** `git clone` télécharge une copie exacte du dépôt, et `cd` vous place à l'intérieur du dossier.

### 2. Ajouter les données brutes (Étape cruciale ⚠️)
Pour respecter les contraintes de GitHub (qui refuse les fichiers > 100 Mo), les fichiers de données volumineux contenant la musique ne sont pas inclus dans ce dépôt web (ils sont bloqués par le `.gitignore`).

1. Récupérez les fichiers de base de données `.csv` fournis pour ce projet (ex: `raw_tracks.csv`, `artists.csv`).
2. Placez-les **obligatoirement** dans le dossier `/bdd/data/` de votre projet local.
> **Pourquoi ?** Notre script Python d'initialisation va aller chercher ces fichiers précis dans ce dossier pour peupler la base de données automatiquement.

### 3. Lancer l'application
Une fois les données en place, la magie de Docker opère. Lancez cette commande à la racine du projet :
```bash
docker compose up --build
```

**Que se passe-t-il pendant le chargement ?**
L'architecture démarre dans un ordre strictement séquentiel et sécurisé :
1. 🗄️ **Base de données (`db`)** : Démarre en premier.
2. 🔄 **Script ETL (`populate-script`)** : Attend que la base soit prête, crée les tables via `init.sql`, lit les CSV, remplit la base, puis s'éteint avec succès.
3. ⚙️ **Backend (`backend`)** : L'API FastAPI attend sagement que le script de peuplement soit terminé à 100% avant de s'allumer.
4. 🖥️ **Frontend (`frontend`)** : L'interface React démarre en dernier, une fois que l'API est prête à lui répondre.

---

## 🌐 5. Accès aux Services

Une fois que le terminal affiche que l'application a démarré, accédez aux outils via votre navigateur :

| Service | Description | URL Locale | Identifiants par défaut |
| :--- | :--- | :--- | :--- |
| **Frontend App** | L'interface utilisateur de l'application | [http://localhost:5173](http://localhost:5173) | - |
| **Backend API** | Documentation interactive Swagger UI | [http://localhost:8000/docs](http://localhost:8000/docs) | - |
| **pgAdmin** | Interface d'administration PostgreSQL | [http://localhost:8080](http://localhost:8080) | Login: `admin@admin.com`<br>Mdp: `admin` |
| **Base de Données** | Accès direct PostgreSQL (Port mappé) | `localhost:5433` | User: `user`<br>Mdp: `password`<br>DB: `mabase` |

---

## 🛑 6. Arrêter et Réinitialiser l'application

Pour stopper les serveurs proprement, placez-vous dans le terminal où tourne Docker, faites `Ctrl+C`, puis tapez :
```bash
docker compose down
```
> **Pourquoi ?** Cette commande arrête les conteneurs, mais **conserve vos données** (la base reste intacte pour la prochaine fois).

**Option "Remise à zéro" :**
Si vous avez modifié les fichiers CSV, ou si vous voulez que la base de données soit entièrement recréée au prochain lancement, tapez :
```bash
docker compose down -v
```
> **Pourquoi ?** L'ajout du `-v` (volumes) détruit les disques durs virtuels de Docker. La base de données est complètement effacée. Au prochain `docker compose up`, le script de peuplement refera tout depuis zéro.