import re
import pandas as pd
from pathlib import Path


def extract_keywords(text):
    if not isinstance(text, str):
        return []
    # split on non-alphanumeric, keep words longer than 2 chars
    words = re.split(r"[^A-Za-z0-9А-Яа-яёЁ]+", text.lower())
    return [w for w in words if len(w) > 2]


def main():
    base = Path(__file__).parent / ".." / "csv"
    projects_path = (base / "projects.csv").resolve()
    users_path = (base / "users.csv").resolve()

    projects = pd.read_csv(projects_path)
    users = pd.read_csv(users_path)

    clicks = []

    for _, user in users.iterrows():
        profile = user.get('short_profile', '') or ''
        role = user.get('role', '') or ''
        user_kws = set(extract_keywords(profile) + extract_keywords(role))

        for _, project in projects.iterrows():
            title = project.get('Name', '') or ''
            desc = project.get('Description', '') or ''
            text = f"{title} {desc}".lower()

            # simple heuristic: if any user keyword appears in project text -> click
            match = any(kw in text for kw in user_kws) if user_kws else False

            if match:
                clicks.append((user['id'], project['id']))

    clicks_df = pd.DataFrame(clicks, columns=['user_id', 'project_id'])
    out_path = (base / "clicks_auto.csv").resolve()
    clicks_df.to_csv(out_path, index=False)

    print(f"Wrote {len(clicks_df)} clicks to {out_path}")


if __name__ == '__main__':
    main()
