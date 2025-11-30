import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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


def main(threshold: float = 0.15, top_k: int = 3):
    base = Path(__file__).parent / ".." / "csv"
    projects_path = (base / "projects.csv").resolve()
    users_path = (base / "users.csv").resolve()

    projects = pd.read_csv(projects_path)
    users = pd.read_csv(users_path)

    project_texts = build_texts(projects)
    user_texts = build_user_texts(users)

    # Fit vectorizer on union to have compatible vocab
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    vectorizer.fit(project_texts + user_texts)

    P = vectorizer.transform(project_texts)
    U = vectorizer.transform(user_texts)

    # Normalize for cosine via dot product
    Pn = normalize(P)
    Un = normalize(U)

    # Compute similarity matrix: users x projects
    sims = Un.dot(Pn.T).toarray()

    clicks = []
    top_examples = []

    for ui, user in users.iterrows():
        sim_row = sims[ui]
        # mark clicks where similarity >= threshold
        clicked_idx = np.where(sim_row >= threshold)[0]
        for pj in clicked_idx:
            clicks.append((user['id'], projects.iloc[pj]['id']))

        # save top-k examples for inspection
        topk_idx = np.argsort(-sim_row)[:top_k]
        topk = [(int(projects.iloc[i]['id']), float(sim_row[i])) for i in topk_idx]
        top_examples.append((int(user['id']), topk))

    clicks_df = pd.DataFrame(clicks, columns=['user_id', 'project_id'])
    out_path = (base / "clicks_tfidf.csv").resolve()
    clicks_df.to_csv(out_path, index=False)

    print(f"Wrote {len(clicks_df)} clicks to {out_path} (threshold={threshold})")

    # Print simple stats
    if len(clicks_df) > 0:
        counts = clicks_df['user_id'].value_counts()
        print(f"Avg clicks per user: {counts.mean():.2f}, median: {counts.median():.1f}")

    print("Sample top-k projects per first 5 users:")
    for ue in top_examples[:5]:
        print(f"User {ue[0]}: {ue[1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.15, help='cosine similarity threshold')
    parser.add_argument('--top-k', type=int, default=3, help='top k examples per user')
    args = parser.parse_args()
    main(threshold=args.threshold, top_k=args.top_k)
