"""
GRU Model - Système de recommandation sémantique
Basé sur l'analyse de l'historique de recherche utilisateur
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from typing import List, Optional


class SemanticGRU(nn.Module):
    """
    Modèle GRU pour prédire le vecteur "envie musicale" 
    à partir d'une séquence de recherches encodées en BERT
    """
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=384):
        super(SemanticGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return torch.nn.functional.normalize(prediction, p=2, dim=1)


class MusicRecommender:
    """
    Système de recommandation musicale complet.
    Singleton pour éviter de recharger le modèle à chaque requête.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.is_ready = False
        
        try:
            self._load_components()
            self.is_ready = True
        except Exception as e:
            print(f"⚠️ Erreur lors de l'initialisation du recommandeur : {e}")
            self.is_ready = False
    
    def _load_components(self):
        """Charge tous les composants du système"""
        print("🔄 Initialisation du système de recommandation...")
        
        base_path = os.path.dirname(__file__)
        systemeia_path = os.path.join(os.path.dirname(base_path), "..", "..", "..", "..", "..", "SystemeIA")
        chunks_path = os.path.join(base_path, "vectors_chunks")
        
        # 1. Encodeur BERT
        print("   → Chargement du modèle BERT...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 2. Vecteurs de la base musicale
        print("   → Chargement des vecteurs musicaux...")
        self.db_vectors = self._load_vectors(base_path, systemeia_path)
        
        # Fonctions utilitaires pour chercher les fichiers
        def find_file(filename):
            # 1. Dossier courant (recommender/)
            path1 = os.path.join(base_path, filename)
            if os.path.exists(path1): return path1
            # 2. Dossier chunks (recommender/vectors_chunks/)
            path2 = os.path.join(chunks_path, filename)
            if os.path.exists(path2): return path2
            # 3. Dossier SystemeIA (R&D)
            path3 = os.path.normpath(os.path.join(systemeia_path, filename))
            if os.path.exists(path3): return path3
            return None

        # 3. Scores d'intérêt (popularité)
        interest_file = find_file("data_interest_109k.pt")
        if not interest_file: raise FileNotFoundError("data_interest_109k.pt introuvable")
        self.db_interests = torch.load(interest_file, map_location='cpu')
        
        # 4. Métadonnées
        meta_file = find_file("data_metadata_109k.pkl")
        if not meta_file: raise FileNotFoundError("data_metadata_109k.pkl introuvable")
        self.df_meta = pd.read_pickle(meta_file)
        
        # 5. Modèle GRU
        print("   → Chargement du modèle GRU...")
        self.model = SemanticGRU(input_dim=384, hidden_dim=256, output_dim=384)
        
        model_file = find_file("modele_gru_109k.pth")
        if not model_file: raise FileNotFoundError("modele_gru_109k.pth introuvable")
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.eval()
        
        print(f"✅ Recommandeur prêt ({len(self.df_meta)} titres disponibles)")
    
    def _load_vectors(self, base_path: str, systemeia_path: str) -> torch.Tensor:
        """
        Charge les vecteurs depuis les chunks ou le fichier original
        """
        # Option 1 : Fichier complet dans SystemeIA
        full_path = os.path.join(systemeia_path, "data_vectors_109k.pt")
        if os.path.exists(full_path):
            return torch.load(full_path, map_location='cpu')
        
        # Option 2 : Chunks découpés
        chunks_dir = os.path.join(base_path, "vectors_chunks")
        metadata_path = os.path.join(chunks_dir, "chunks_metadata.pt")
        
        if os.path.exists(metadata_path):
            print("   → Reconstruction depuis les chunks...")
            metadata = torch.load(metadata_path, map_location='cpu')
            
            chunks = []
            for i in range(metadata["num_chunks"]):
                chunk_path = os.path.join(chunks_dir, f"vectors_chunk_{i}.pt")
                chunk = torch.load(chunk_path, map_location='cpu')
                chunks.append(chunk)
            
            return torch.cat(chunks, dim=0)
        
        # Option 3 : Fichier local direct
        local_path = os.path.join(base_path, "data_vectors_109k.pt")
        if os.path.exists(local_path):
            return torch.load(local_path, map_location='cpu')
        
        raise FileNotFoundError("Aucune source de vecteurs musicaux trouvée !")
    
    def predict(self, search_history: List[str], top_k: int = 10) -> List[int]:
        """
        Génère des recommandations basées sur l'historique de recherche.
        
        Args:
            search_history: Liste des recherches (ordre chronologique, ancien → récent)
            top_k: Nombre de recommandations à retourner
            
        Returns:
            Liste de track_id recommandés
        """
        if not self.is_ready:
            return []
        
        if not search_history:
            return []
        
        with torch.no_grad():
            # 1. Encodage BERT des recherches
            input_vectors = self.encoder.encode(search_history, convert_to_tensor=True)
            input_tensor = input_vectors.unsqueeze(0)  # [1, seq_len, 384]
            
            # 2. Prédiction du vecteur "envie"
            user_vibe_vector = self.model(input_tensor)  # [1, 384]
            
            # 3. Similarité cosinus avec toute la base
            similarities = torch.nn.functional.cosine_similarity(
                user_vibe_vector, 
                self.db_vectors
            )
            
            # 4. Score hybride (70% sémantique + 30% popularité)
            final_scores = (similarities * 0.7) + (self.db_interests * 0.3)
            
            # 5. Top-K
            top_indices = torch.topk(final_scores, top_k).indices.tolist()
            
            # 6. Extraction des track_id
            recommendations = [
                int(self.df_meta.iloc[idx]['track_id']) 
                for idx in top_indices
            ]
        
        return recommendations


# Instance globale (chargée au premier import)
_recommender: Optional[MusicRecommender] = None

def get_recommender() -> MusicRecommender:
    """Retourne le singleton du recommandeur"""
    global _recommender
    if _recommender is None:
        _recommender = MusicRecommender()
    return _recommender
