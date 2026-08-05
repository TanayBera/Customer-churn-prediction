"""
SETUP SCRIPT — Run this ONCE before anything else
This creates your project folder structure automatically
"""

import os

# All folders to create
folders = [
    'data/raw',
    'data/processed',
    'notebooks',
    'src',
    'models',
    'app',
    'tests',
    'docs'
]

print('Creating project structure...')
print()

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f'  ✅ Created: {folder}/')

print()
print('Project structure ready!')
print()
print('Your folder structure:')
print('''
churn-prediction/
│
├── data/
│   ├── raw/          ← NEVER touch files here (sacred original data)
│   └── processed/    ← Your cleaned data goes here
│
├── notebooks/        ← Run your jupyter notebooks here
│   ├── 01_data_first_look.ipynb     ← DAY 1
│   ├── 02_deep_exploration.ipynb    ← DAY 2
│   ├── 03_feature_engineering.ipynb ← DAY 3
│   ├── 04_model_building.ipynb      ← DAY 5-6
│   └── 05_model_evaluation.ipynb    ← DAY 6
│
├── src/              ← Clean python code (not notebooks)
├── models/           ← Your saved trained models (.pkl files)
├── app/              ← Streamlit dashboard code
├── tests/            ← Tests for your code
├── docs/             ← Charts, diagrams, notes
│
├── requirements.txt  ← List of all libraries needed
└── README.md         ← Project description for GitHub
''')
print()
print('NEXT STEP: Open notebooks/01_data_first_look.ipynb')
print('Run: jupyter notebook')
