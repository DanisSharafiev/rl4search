import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `src` package can be imported when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDER_MODEL_NAME
from src.utils.similarity_metrics import cosine_similarity


def get_model(name: str = EMBEDDER_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_texts_local(texts, model: SentenceTransformer):
    # keep behavior similar to project helper: return numpy array
    emb = model.encode(texts, convert_to_tensor=True)
    return emb.cpu().numpy()


def build_texts(projects: pd.DataFrame):
    texts = []
    for _, p in projects.iterrows():
        title = p.get('Name', '') or ''
        desc = p.get('Description', '') or ''
        texts.append(f"{title} {desc}")
    return texts


def build_user_texts(users: pd.DataFrame):
    texts = []
    for _, u in users.iterrows():
        role = u.get('role', '') or ''
        profile = u.get('short_profile', '') or ''
        texts.append(f"{role} {profile}")
    return texts


def main(threshold: float = 0.3, top_k: int = 3, batch_size: int = 64):
    base = Path(__file__).parent / ".." / "csv"
    projects_path = (base / "projects.csv").resolve()
    users_path = (base / "users.csv").resolve()

    projects = pd.read_csv(projects_path)
    users = pd.read_csv(users_path)

    project_texts = build_texts(projects)
    user_texts = build_user_texts(users)

    # Load model and compute embeddings
    model = get_model()

    print("Embedding projects...")
    P = embed_texts_local(project_texts, model)
    print("Embedding users...")
    U = embed_texts_local(user_texts, model)

    # Normalize to unit vectors
    def normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    Pn = normalize_rows(P)
    Un = normalize_rows(U)

    # Compute similarity matrix in a memory-friendly way
    sims = Un.dot(Pn.T)

    clicks = []
    top_examples = []

    for ui, user in users.iterrows():
        sim_row = sims[ui]
        clicked_idx = np.where(sim_row >= threshold)[0]
        for pj in clicked_idx:
            clicks.append((user['id'], projects.iloc[pj]['id']))

        topk_idx = np.argsort(-sim_row)[:top_k]
        topk = [(int(projects.iloc[i]['id']), float(sim_row[i])) for i in topk_idx]
        top_examples.append((int(user['id']), topk))

    clicks_df = pd.DataFrame(clicks, columns=['user_id', 'project_id'])
    out_path = (base / "clicks_embed.csv").resolve()
    clicks_df.to_csv(out_path, index=False)

    print(f"Wrote {len(clicks_df)} clicks to {out_path} (threshold={threshold})")
    if len(clicks_df) > 0:
        counts = clicks_df['user_id'].value_counts()
        print(f"Avg clicks per user: {counts.mean():.2f}, median: {counts.median():.1f}")

    print("Sample top-k projects per first 5 users:")
    for ue in top_examples[:5]:
        print(f"User {ue[0]}: {ue[1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    main(threshold=args.threshold, top_k=args.top_k)
