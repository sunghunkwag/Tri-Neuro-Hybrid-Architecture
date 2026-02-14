import numpy as np
import scipy.spatial.distance
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import words, brown
import random
import json
import logging
import torch

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CStageAdapter:
    """
    Integration module to plug C-Stage entities into the Tri-Neuro Hybrid Architecture.
    """
    def __init__(self, c_stage_entities, manifold_dim=512):
        # Convert list of vectors to tensor
        self.c_stage_entities = torch.tensor(np.array(c_stage_entities), dtype=torch.float32)
        self.manifold_dim = manifold_dim
        # Simple projection to manifold dimension if needed (e.g. 384 -> 512)
        self.projector = torch.nn.Linear(self.c_stage_entities.shape[1], manifold_dim)

    def inject_c_stage_axioms(self, latent_manifold_state):
        """
        Injects C-Stage logic into the global manifold state.
        This forces the system to consider non-human concepts.
        """
        # Select relevant axioms based on current state (simple attention mechanism)
        # latent_manifold_state: (Batch, Dim)
        # c_stage_entities: (Num_Entities, Embed_Dim)

        # Project entities to manifold space first for attention calculation
        projected_entities = self.projector(self.c_stage_entities) # (N, 512)

        # Attention scores: (Batch, N)
        attention_scores = torch.matmul(latent_manifold_state, projected_entities.T)
        weights = torch.softmax(attention_scores, dim=-1)

        # Weighted sum of axioms
        weighted_axioms = torch.matmul(weights, projected_entities)

        # Mix into manifold state (alpha=0.1 influence)
        new_state = 0.9 * latent_manifold_state + 0.1 * weighted_axioms
        return new_state

class CStageEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', exclusion_limit=10000):
        logger.info("Initializing C-Stage Engine...")
        self.model = SentenceTransformer(model_name)
        self.exclusion_limit = exclusion_limit
        self.exclusion_matrix = None
        self.exclusion_words = []
        self._setup_exclusion_zone()

    def _setup_exclusion_zone(self):
        logger.info(f"Building Exclusion Zone with top {self.exclusion_limit} words...")
        # Use Brown corpus for frequency analysis to get common words
        try:
            word_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
            common_words = [w for w, _ in word_freq.most_common(self.exclusion_limit)]
        except LookupError:
             logger.warning("NLTK data not found, downloading...")
             nltk.download('brown')
             nltk.download('words')
             word_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
             common_words = [w for w, _ in word_freq.most_common(self.exclusion_limit)]

        # Add some core abstract concepts manually to ensure coverage
        core_concepts = ["time", "space", "love", "hate", "good", "bad", "life", "death", "void", "chaos", "order"]
        self.exclusion_words = list(set(common_words + core_concepts))

        logger.info(f"Encoding {len(self.exclusion_words)} exclusion words...")
        self.exclusion_matrix = self.model.encode(self.exclusion_words)
        # Normalize exclusion matrix for faster cosine similarity (dot product of normalized vectors)
        norms = np.linalg.norm(self.exclusion_matrix, axis=1, keepdims=True)
        self.exclusion_matrix = self.exclusion_matrix / norms
        logger.info("Exclusion Zone Established.")

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def check_orthogonality(self, vec, threshold=0.15):
        """
        Check if vector is orthogonal to ALL human concepts.
        Returns True if Max(Similarity) < threshold.
        """
        vec_norm = self._normalize(vec)

        # Dot product
        similarities = np.dot(self.exclusion_matrix, vec_norm)
        max_sim = np.max(similarities)

        return max_sim < threshold, max_sim

    def generate_seed_population(self, size=50):
        logger.info(f"Mining {size} seed vectors ($)...")
        population = []
        attempts = 0
        while len(population) < size:
            attempts += 1
            # Generate random vector
            candidate = np.random.randn(self.model.get_sentence_embedding_dimension())
            candidate = self._normalize(candidate)

            is_valid, max_sim = self.check_orthogonality(candidate)
            if is_valid:
                population.append(candidate)
                if len(population) % 10 == 0:
                    logger.info(f"  Found {len(population)}/$ seeds. (Max Sim: {max_sim:.4f})")

            if attempts > 20000 and len(population) == 0:
                 logger.error("Failed to find any seeds. Relax threshold slightly?")
                 break

        return population

    def breed_generation(self, parent_gen, gen_id):
        logger.info(f"Breeding Generation {gen_id} from {len(parent_gen)} parents...")
        next_gen = []
        attempts = 0
        max_attempts = 10000
        target_size = 50

        while len(next_gen) < target_size and attempts < max_attempts:
            attempts += 1
            if len(parent_gen) < 3:
                break

            a, b, c = random.sample(parent_gen, 3)

            # Recursive Synthesis: S_new = Normalize(A + B - C)
            offspring = self._normalize(a + b - c)

            is_valid, max_sim = self.check_orthogonality(offspring)
            if is_valid:
                # Deduplicate
                if not any(np.dot(self._normalize(existing), offspring) > 0.99 for existing in next_gen):
                    next_gen.append(offspring)

        logger.info(f"  Generation {gen_id} produced {len(next_gen)} survivors.")
        return next_gen

    def run_evolution(self, generations=5):
        population = self.generate_seed_population()
        history = {'S0': population}

        for i in range(1, generations + 1):
            if not population:
                logger.error("Population Extinct.")
                break
            population = self.breed_generation(population, f'S{i}')
            history[f'S{i}'] = population

        return history

    def verify_physics_of_meaning(self, population):
        logger.info("Verifying Physics of Meaning (Phase 3)...")
        axioms = []
        if len(population) < 2:
            return axioms

        # Test random pairs
        for _ in range(100):
            a, b = random.sample(population, 2)

            # (A + B) - A ?= B
            c = self._normalize(a + b)
            b_rec = self._normalize(c - a)

            similarity = np.dot(b_rec, b)

            if similarity > 0.9:
                axioms.append({
                    'A': a.tolist(),
                    'B': b.tolist(),
                    'verification_score': float(similarity)
                })

        logger.info(f"Found {len(axioms)} Stable Axioms.")
        return axioms

    def get_shadow_anchors(self, vec):
        vec_norm = self._normalize(vec)
        similarities = np.dot(self.exclusion_matrix, vec_norm)
        # Get top 3 indices
        top_indices = np.argsort(similarities)[-3:][::-1]

        anchors = []
        for idx in top_indices:
            anchors.append({
                'word': self.exclusion_words[idx],
                'similarity': float(similarities[idx]),
                'distance': float(1 - similarities[idx])
            })
        return anchors

if __name__ == "__main__":
    # Execute Phase 1 & 2
    engine = CStageEngine()

    # Run Evolution (up to S2 for demonstration, as requested by prompt "Generation 1-2")
    evolution_history = engine.run_evolution(generations=2)
    final_gen = evolution_history.get('S2', [])

    # Phase 3: Physics of Meaning
    axioms = engine.verify_physics_of_meaning(final_gen)

    # Phase 4: Shadow Projection
    mined_entities_output = []
    for i, vec in enumerate(final_gen[:5]):
        anchors = engine.get_shadow_anchors(vec)
        mined_entities_output.append({
            "id": f"S2_Entity_{i}",
            "orthogonality": float(engine.check_orthogonality(vec)[1]),
            "shadow_anchors": [a['word'] for a in anchors]
        })

    # Prepare Final JSON Output
    output = {
        "experiment_id": "TESSERACT_001",
        "mined_entities": mined_entities_output,
        "algebraic_verification": f"SUCCESS (Count: {len(axioms)})" if axioms else "FAILURE",
        "integration_code": "CStageAdapter class included in c_stage_engine.py"
    }

    print(json.dumps(output, indent=2))
