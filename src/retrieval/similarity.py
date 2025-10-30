"""
Similarity calculation and retrieval system
"""

import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class MusicRetrieval:
    """
    Music retrieval system using cosine similarity
    """

    def __init__(self, encoder=None):
        """
        Initialize retrieval system

        Args:
            encoder: AudioEncoder instance
        """
        self.encoder = encoder
        self.reference_embeddings = None
        self.reference_paths = None

    def build_reference_database(self, reference_dir: str, cache_file: str = None):
        """
        Build reference database from directory

        Args:
            reference_dir: Directory containing reference music
            cache_file: Path to cache embeddings (optional)
        """
        print(f"Building reference database from {reference_dir}")

        if cache_file and Path(cache_file).exists():
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.reference_embeddings = data['embeddings']
                self.reference_paths = data['paths']
        else:
            # Encode all reference files
            embeddings_dict = self.encoder.encode_directory(reference_dir)

            self.reference_paths = list(embeddings_dict.keys())
            self.reference_embeddings = np.array([embeddings_dict[path] for path in self.reference_paths])

            # Cache if requested
            if cache_file:
                print(f"Caching embeddings to {cache_file}")
                Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.reference_embeddings,
                        'paths': self.reference_paths
                    }, f)

        print(f"Reference database built with {len(self.reference_paths)} files")

    def calculate_similarity(self, query_embedding: np.ndarray,
                            reference_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            query_embedding: Query embedding
            reference_embedding: Reference embedding

        Returns:
            Cosine similarity score
        """
        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(reference_embedding.shape) == 1:
            reference_embedding = reference_embedding.reshape(1, -1)

        similarity = cosine_similarity(query_embedding, reference_embedding)[0, 0]
        return float(similarity)

    def find_most_similar(self, query_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar reference tracks to query

        Args:
            query_path: Path to query audio file
            top_k: Number of top matches to return

        Returns:
            List of tuples (reference_path, similarity_score)
        """
        if self.reference_embeddings is None:
            raise ValueError("Reference database not built. Call build_reference_database() first.")

        if len(self.reference_paths) == 0:
            raise ValueError("Reference database is empty. No audio files found in reference directory.")

        # Encode query
        query_embedding = self.encoder.encode_audio(query_path)
        query_embedding = query_embedding.reshape(1, -1)

        # Calculate similarities with all references
        similarities = cosine_similarity(query_embedding, self.reference_embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return paths and scores
        results = [(self.reference_paths[idx], float(similarities[idx]))
                   for idx in top_indices]

        return results

    def retrieve_for_targets(self, target_dir: str, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve similar references for all target files

        Args:
            target_dir: Directory containing target music
            top_k: Number of matches per target

        Returns:
            Dictionary mapping target paths to list of (reference_path, score) tuples
        """
        target_dir = Path(target_dir)
        target_files = []

        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            target_files.extend(target_dir.glob(f"*{ext}"))

        results = {}

        for target_file in target_files:
            print(f"\nRetrieving for: {target_file.name}")
            similar = self.find_most_similar(str(target_file), top_k=top_k)

            results[str(target_file)] = similar

            # Print top results
            print(f"Top {min(3, len(similar))} matches:")
            for ref_path, score in similar[:3]:
                print(f"  {Path(ref_path).name}: {score:.4f}")

        return results

    def save_results(self, results: Dict, output_file: str):
        """
        Save retrieval results to file

        Args:
            results: Dictionary of retrieval results
            output_file: Path to output file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to {output_file}")

        # Also save as text for readability
        text_file = output_file.with_suffix('.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            for target_path, matches in results.items():
                f.write(f"\nTarget: {Path(target_path).name}\n")
                f.write("=" * 80 + "\n")
                for i, (ref_path, score) in enumerate(matches, 1):
                    f.write(f"{i}. {Path(ref_path).name}: {score:.4f}\n")
                f.write("\n")

        print(f"Text results saved to {text_file}")


if __name__ == "__main__":
    from audio_encoder import AudioEncoder

    # Initialize
    encoder = AudioEncoder()
    retrieval = MusicRetrieval(encoder)

    # Build reference database
    reference_dir = "./fundwotsai/Deep_MIR_hw2/referecne_music_list_60s"
    retrieval.build_reference_database(
        reference_dir,
        cache_file="./outputs/reference_embeddings.pkl"
    )

    # Retrieve for all targets
    target_dir = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s"
    results = retrieval.retrieve_for_targets(target_dir, top_k=5)

    # Save results
    retrieval.save_results(results, "./outputs/retrieved/retrieval_results.pkl")
