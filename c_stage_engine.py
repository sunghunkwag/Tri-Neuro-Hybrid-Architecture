"""
Tesseract-Synthesis V2: The Operator Era
---------------------------------------
Upgrades "geometric mining" into "meta-evolutionary synthesis":

1) Human semantic space => continuous density model (GMM) in *latent* space
2) Orthogonality => maximize information divergence (high NLL under human density)
3) Vector breeding => evolve TRANSFORMATION OPERATORS (learnable modules)
4) Learnable manifold => lightweight Autoencoder (AE) trained on human embeddings
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import brown

try:
    from sklearn.mixture import GaussianMixture
except Exception as e:
    raise ImportError(
        "scikit-learn is required for GaussianMixture. Install: pip install scikit-learn"
    ) from e


# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tesseract_v2")


# -------------------------
# Utilities
# -------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Autoencoder Manifold
# -------------------------
class TinyAutoencoder(nn.Module):
    def __init__(self, in_dim: int, z_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, z_dim),
        )

        self.dec = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# -------------------------
# Operator Genome
# -------------------------
class OperatorModule(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class OperatorGenome:
    state_dict: Dict[str, torch.Tensor]

    @staticmethod
    def from_model(model: nn.Module) -> "OperatorGenome":
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return OperatorGenome(sd)

    def into_model(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict, strict=True)

    def mutate(self, sigma: float = 0.05, p: float = 0.10) -> "OperatorGenome":
        new_sd = {}
        for k, v in self.state_dict.items():
            vv = v.clone()
            if vv.dtype.is_floating_point:
                mask = (torch.rand_like(vv) < p).float()
                noise = torch.randn_like(vv) * sigma
                vv = vv + mask * noise
            new_sd[k] = vv
        return OperatorGenome(new_sd)

    @staticmethod
    def crossover(a: "OperatorGenome", b: "OperatorGenome", p: float = 0.5) -> "OperatorGenome":
        new_sd = {}
        for k in a.state_dict.keys():
            va = a.state_dict[k]
            vb = b.state_dict[k]
            if va.dtype.is_floating_point:
                mask = (torch.rand_like(va) < p).float()
                vc = mask * va + (1.0 - mask) * vb
            else:
                vc = va
            new_sd[k] = vc
        return OperatorGenome(new_sd)


# -------------------------
# Core Engine
# -------------------------
class TesseractV2Engine:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        exclusion_limit: int = 10000,
        z_dim: int = 128,
        gmm_components: int = 32,
        device: str | None = None,
    ):
        logger.info("Initializing Tesseract V2 Engine...")
        self.embedder = SentenceTransformer(model_name)
        self.exclusion_limit = exclusion_limit
        self.z_dim = z_dim
        self.gmm_components = gmm_components

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.exclusion_words: List[str] = []
        self.X_human: np.ndarray | None = None
        self.Z_human: np.ndarray | None = None

        self.ae: TinyAutoencoder | None = None
        self.gmm: GaussianMixture | None = None
        self.X_human_norm: np.ndarray | None = None

    def build_human_pool(self) -> None:
        logger.info(f"Building human pool with top {self.exclusion_limit} words...")
        try:
            word_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
        except LookupError:
            logger.warning("NLTK data not found. Downloading brown corpus...")
            nltk.download("brown")
            word_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())

        common_words = [w for w, _ in word_freq.most_common(self.exclusion_limit)]
        core = [
            "time", "space", "love", "hate", "good", "bad", "life", "death", "void",
            "chaos", "order", "god", "truth", "meaning", "mind", "matter", "self",
        ]
        self.exclusion_words = list(dict.fromkeys(common_words + core))
        logger.info(f"Human pool size: {len(self.exclusion_words)}")

    def embed_human_pool(self) -> None:
        if not self.exclusion_words:
            self.build_human_pool()
        logger.info("Encoding human pool embeddings...")
        X = self.embedder.encode(self.exclusion_words, show_progress_bar=True)
        X = np.asarray(X, dtype=np.float32)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.X_human = X
        self.X_human_norm = Xn
        logger.info(f"X_human: {self.X_human.shape}")

    def train_autoencoder(self, epochs: int = 8, batch_size: int = 256, lr: float = 2e-3, weight_decay: float = 1e-4) -> None:
        assert self.X_human is not None
        embed_dim = int(self.X_human.shape[1])
        self.ae = TinyAutoencoder(embed_dim, self.z_dim).to(self.device)
        opt = torch.optim.AdamW(self.ae.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.tensor(self.X_human, dtype=torch.float32, device=self.device)
        n = X.shape[0]

        logger.info(f"Training AE: epochs={epochs}, batch={batch_size}")
        self.ae.train()
        for ep in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx]
                x_hat, z = self.ae(xb)
                recon = F.mse_loss(x_hat, xb)
                z_reg = (z.pow(2).mean()) * 1e-3
                loss = recon + z_reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X).detach().cpu().numpy().astype(np.float32)
        self.Z_human = Z
        logger.info(f"Z_human: {self.Z_human.shape}")

    def fit_gmm(self, reg_covar: float = 1e-5, max_iter: int = 200) -> None:
        assert self.Z_human is not None
        logger.info(f"Fitting GMM: components={self.gmm_components}")
        gmm = GaussianMixture(n_components=self.gmm_components, covariance_type="full", reg_covar=reg_covar, max_iter=max_iter, random_state=42)
        gmm.fit(self.Z_human)
        self.gmm = gmm
        ll = gmm.score_samples(self.Z_human)
        logger.info(f"GMM fitted. LL mean={ll.mean():.3f}")

    def _fitness(self, op: OperatorModule, samples_per_fit: int = 512, alpha_decode_reencode: float = 0.25, beta_norm: float = 0.02) -> float:
        assert self.ae is not None and self.gmm is not None
        op.eval()
        with torch.no_grad():
            z0 = torch.randn(samples_per_fit, self.z_dim, device=self.device)
            z1 = op(z0)
            z1_np = z1.detach().cpu().numpy().astype(np.float32)
            ll = self.gmm.score_samples(z1_np)
            nll = -ll.mean()
            x_syn = self.ae.decode(z1)
            z_rec = self.ae.encode(x_syn)
            cons = float(F.mse_loss(z_rec, z1).item())
            norm_pen = float(z1.pow(2).mean().item())
        return float(nll - alpha_decode_reencode * cons - beta_norm * norm_pen)

    def evolve_operators(self, pop_size: int = 24, generations: int = 30, elite_k: int = 6, samples_per_fit: int = 512) -> List[OperatorGenome]:
        assert self.ae is not None and self.gmm is not None
        base_model = OperatorModule(self.z_dim).to(self.device)
        population = [OperatorGenome.from_model(OperatorModule(self.z_dim).to(self.device)) for _ in range(pop_size)]

        for g in range(1, generations + 1):
            scored = []
            for genome in population:
                genome.into_model(base_model)
                fit = self._fitness(base_model, samples_per_fit=samples_per_fit)
                scored.append((fit, genome))
            scored.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"  Gen {g}/{generations} best={scored[0][0]:.4f}")

            elites = [gen for _, gen in scored[:elite_k]]
            new_pop = elites.copy()
            while len(new_pop) < pop_size:
                if random.random() < 0.6 and len(elites) >= 2:
                    a, b = random.sample(elites, 2)
                    child = OperatorGenome.crossover(a, b)
                else:
                    child = elites[0]
                child = child.mutate()
                new_pop.append(child)
            population = new_pop

        final_scored = []
        for genome in population:
            genome.into_model(base_model)
            fit = self._fitness(base_model, samples_per_fit=samples_per_fit)
            final_scored.append((fit, genome))
        final_scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in final_scored]

    def mine_entities(self, best_genome: OperatorGenome, count: int = 8, anchors_k: int = 5) -> List[Dict]:
        op = OperatorModule(self.z_dim).to(self.device)
        best_genome.into_model(op)
        op.eval()
        entities = []
        with torch.no_grad():
            z0 = torch.randn(count, self.z_dim, device=self.device)
            z1 = op(z0)
            x_syn = self.ae.decode(z1).detach().cpu().numpy().astype(np.float32)

        x_syn_n = x_syn / (np.linalg.norm(x_syn, axis=1, keepdims=True) + 1e-12)
        z1_np = z1.detach().cpu().numpy().astype(np.float32)
        ll = self.gmm.score_samples(z1_np)
        nll = (-ll).tolist()

        for i in range(count):
            sims = np.dot(self.X_human_norm, x_syn_n[i])
            top_idx = np.argsort(sims)[-anchors_k:][::-1]
            anchors = [
                {"word": self.exclusion_words[int(j)], "similarity": float(sims[int(j)]), "distance": float(1.0 - sims[int(j)])}
                for j in top_idx
            ]
            entities.append({
                "id": f"V2_Entity_{i}",
                "latent_nll": float(nll[i]),
                "shadow_anchors": [a["word"] for a in anchors],
                "anchors_detail": anchors,
            })
        return entities

    def run(self, ae_epochs: int = 8, gmm_components: int = 32, evo_gens: int = 30, out_entities: int = 8) -> Dict:
        self.gmm_components = gmm_components
        self.build_human_pool()
        self.embed_human_pool()
        self.train_autoencoder(epochs=ae_epochs)
        self.fit_gmm()
        genomes = self.evolve_operators(generations=evo_gens)
        mined = self.mine_entities(genomes[0], count=out_entities)
        return {
            "experiment_id": "TESSERACT_V2_OPERATOR_ERA",
            "human_pool_size": len(self.exclusion_words),
            "latent_dim": self.z_dim,
            "mined_entities": mined,
        }

if __name__ == "__main__":
    set_seed(42)
    engine = TesseractV2Engine(model_name="all-MiniLM-L6-v2", exclusion_limit=10000, z_dim=128, gmm_components=32)
    result = engine.run(ae_epochs=6, evo_gens=20, out_entities=6)
    print(json.dumps(result, indent=2))
