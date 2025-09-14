import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Tuple, Dict
import time


class MovieEmbeddingSystem:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', device: str = 'auto'):
        """
        Sistema de embeddings para películas

        Args:
            model_name: Nombre del modelo de Hugging Face
            device: 'auto', 'cuda', 'cpu' o dispositivo específico como 'cuda:0'
        """
        # Detectar dispositivo automáticamente
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🔥 GPU detectada: {torch.cuda.get_device_name()}")
                print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = 'cpu'
                print("⚠️ GPU no disponible, usando CPU")

        print(f"🎬 Cargando modelo: {model_name}")
        print(f"🖥️ Dispositivo: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.movie_embeddings = None
        self.movies_df = None

    def prepare_movie_text(self, row: pd.Series) -> str:
        """
        Combina información de la película en un texto para embedding
        """
        # Combinar diferentes campos (ajusta según tu dataset)
        parts = []

        if 'title' in row and pd.notna(row['title']):
            parts.append(f"Título: {row['title']}")

        if 'genre' in row and pd.notna(row['genre']):
            parts.append(f"Género: {row['genre']}")

        if 'overview' in row and pd.notna(row['overview']):
            parts.append(f"Sinopsis: {row['overview']}")

        if 'director' in row and pd.notna(row['director']):
            parts.append(f"Director: {row['director']}")

        if 'cast' in row and pd.notna(row['cast']):
            parts.append(f"Reparto: {row['cast']}")

        if 'keywords' in row and pd.notna(row['keywords']):
            parts.append(f"Palabras clave: {row['keywords']}")

        return " | ".join(parts)

    def create_embeddings(self, movies_df: pd.DataFrame, save_path: str = None, batch_size: int = None) -> np.ndarray:
        """
        Crea embeddings para todo el dataset de películas

        Args:
            movies_df: DataFrame con las películas
            save_path: Ruta para guardar los embeddings
            batch_size: Tamaño del lote (auto-detectado según GPU/CPU)
        """
        print(f"🔄 Preparando texto para {len(movies_df)} películas...")

        # Auto-detectar batch_size óptimo según dispositivo
        if batch_size is None:
            if self.device.startswith('cuda'):
                import torch
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 8:
                    batch_size = 128  # GPU con >= 8GB
                elif gpu_memory >= 4:
                    batch_size = 64  # GPU con >= 4GB
                else:
                    batch_size = 32  # GPU con < 4GB
                print(f"🚀 Batch size optimizado para GPU: {batch_size}")
            else:
                batch_size = 16  # CPU
                print(f"🐌 Batch size para CPU: {batch_size}")

        # Preparar textos
        movie_texts = []
        for idx, row in movies_df.iterrows():
            text = self.prepare_movie_text(row)
            movie_texts.append(text)

        print(f"📝 Ejemplo de texto preparado:")
        print(f"   {movie_texts[0][:200]}...")

        # Crear embeddings con configuración optimizada
        print(f"🧠 Generando embeddings en {self.device.upper()}...")
        start_time = time.time()

        # Configuración adicional para GPU
        encode_kwargs = {
            'show_progress_bar': True,
            'batch_size': batch_size,
            'convert_to_numpy': True,  # Más eficiente para almacenamiento
        }

        # Si es GPU, añadir configuraciones específicas
        if self.device.startswith('cuda'):
            encode_kwargs['normalize_embeddings'] = True  # Normalización en GPU

        embeddings = self.model.encode(movie_texts, **encode_kwargs)

        elapsed = time.time() - start_time
        movies_per_second = len(movies_df) / elapsed
        print(f"✅ Embeddings creados en {elapsed:.2f} segundos")
        print(f"⚡ Velocidad: {movies_per_second:.1f} películas/segundo")
        print(f"📊 Forma: {embeddings.shape}")

        # Limpiar memoria GPU si es necesario
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
            print("🧹 Cache GPU limpiado")

        # Guardar
        self.movie_embeddings = embeddings
        self.movies_df = movies_df

        if save_path:
            self.save_embeddings(save_path)

        return embeddings

    def save_embeddings(self, path: str):
        """Guarda embeddings y metadatos"""
        data = {
            'embeddings': self.movie_embeddings,
            'movies_df': self.movies_df,
            'model_name': self.model.get_sentence_embedding_dimension()
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Embeddings guardados en: {path}")

    def load_embeddings(self, path: str):
        """Carga embeddings guardados"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.movie_embeddings = data['embeddings']
        self.movies_df = data['movies_df']
        print(f"📂 Embeddings cargados desde: {path}")

    def find_similar_movies(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Encuentra películas similares a una consulta
        """
        if self.movie_embeddings is None:
            raise ValueError("Primero debes crear o cargar embeddings")

        # Crear embedding de la consulta
        query_embedding = self.model.encode([query])

        # Calcular similitudes
        similarities = cosine_similarity(query_embedding, self.movie_embeddings)[0]

        # Obtener top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            movie_title = self.movies_df.iloc[idx]['title']
            similarity = similarities[idx]
            results.append((idx, movie_title, similarity))

        return results

    def find_similar_to_movie(self, movie_id: int, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Encuentra películas similares a una película específica
        """
        if self.movie_embeddings is None:
            raise ValueError("Primero debes crear o cargar embeddings")

        # Embedding de la película de referencia
        movie_embedding = self.movie_embeddings[movie_id].reshape(1, -1)

        # Calcular similitudes
        similarities = cosine_similarity(movie_embedding, self.movie_embeddings)[0]

        # Obtener top k (excluyendo la película original)
        top_indices = np.argsort(similarities)[::-1][1:top_k + 1]

        results = []
        for idx in top_indices:
            movie_title = self.movies_df.iloc[idx]['title']
            similarity = similarities[idx]
            results.append((idx, movie_title, similarity))

        return results


# Ejemplo de uso optimizado para GPU
def ejemplo_uso():
    """Ejemplo de cómo usar el sistema con GPU"""

    # Datos de ejemplo (reemplaza con tu dataset real)
    movies_data = {
        'title': ['The Matrix', 'Inception', 'Toy Story', 'The Dark Knight', 'Finding Nemo'],
        'genre': ['Sci-Fi', 'Sci-Fi', 'Animation', 'Action', 'Animation'],
        'overview': [
            'A computer hacker learns about the true nature of reality',
            'A thief who steals corporate secrets through dream-sharing technology',
            'A cowboy doll is profoundly threatened when a new spaceman toy supplants him',
            'Batman fights against the Joker who wants to turn Gotham into chaos',
            'A clown fish searches for his missing son across the ocean'
        ],
        'director': ['Wachowski', 'Nolan', 'Lasseter', 'Nolan', 'Stanton'],
        'cast': ['Keanu Reeves', 'Leonardo DiCaprio', 'Tom Hanks', 'Christian Bale', 'Albert Brooks']
    }

    movies_df = pd.DataFrame(movies_data)

    # Crear sistema con GPU (auto-detecta)
    embedding_system = MovieEmbeddingSystem(device='auto')

    # Para GPU específica: MovieEmbeddingSystem(device='cuda:0')
    # Para forzar CPU: MovieEmbeddingSystem(device='cpu')

    # Crear embeddings (batch_size se optimiza automáticamente)
    embeddings = embedding_system.create_embeddings(movies_df)

    # Buscar películas similares
    print("\n🔍 Búsqueda por texto:")
    results = embedding_system.find_similar_movies("ciencia ficción", top_k=3)
    for idx, title, similarity in results:
        print(f"   {title}: {similarity:.3f}")

    print("\n🎭 Películas similares a 'The Matrix':")
    results = embedding_system.find_similar_to_movie(0, top_k=3)
    for idx, title, similarity in results:
        print(f"   {title}: {similarity:.3f}")


if __name__ == "__main__":
    ejemplo_uso()