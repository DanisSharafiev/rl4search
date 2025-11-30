import pandas as pd

projects = pd.read_csv('../csv/projects.csv')
users = pd.read_csv('../csv/users.csv')

clicks = []

for _, user in users.iterrows():
    print(user['role'], '\n', user['short_profile'])
    print()
    for _, project in projects.iterrows():
        print(f"Project ID: {project['id']}")
        print(f"Title: {project['Name']}")
        print(f"Description: {project['Description']}")
        click = input("Do you want to click on this project? (yes/no): DEFAULT no:")
        if click.lower() == 'yes':
            clicks.append((user['id'], project['id']))
        print()

clicks = pd.DataFrame(clicks, columns=['user_id', 'project_id'])
clicks.to_csv('../csv/clicks.csv', index=False)
