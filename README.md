# Auto-ML Adaptatif : Recommandation Dynamique de Modèles

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![River](https://img.shields.io/badge/Library-River-green.svg)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-teal.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

Ce projet propose un système **Auto-ML adaptatif** développé dans le cadre d'une étude sur les stratégies d'exploration de données. Il vise à résoudre les problématiques de *batch learning* traditionnel face aux flux de données non stationnaires (*Concept Drift*) dans un environnement de production.

Le système orchestre trois classifieurs incrémentaux en apprentissage simultané et sélectionne dynamiquement le meilleur modèle via un algorithme de bandit multi-bras $\epsilon$-greedy. L'application principale porte sur la détection d'intrusions réseau avec le dataset **UNSW-NB15**.

## 🚀 Fonctionnalités Clés

- **Apprentissage en ligne (Streaming ML)** : Entraînement incrémental sur flux continu sans réentraînement complet.
- **Sélection dynamique de modèles** : Algorithme Bandit multi-bras ($\epsilon$-greedy, $\epsilon=0.1$) arbitrant entre Hoeffding Tree, KNN et SGD.
- **Détection de Dérive (Concept Drift)** : Intégration en parallèle des détecteurs **ADWIN** et **Page-Hinkley** pour identifier les dérives graduelles et abruptes.
- **Observabilité Temps Réel** : 
  - API REST développée avec FastAPI exposant les métriques.
  - Tableau de bord interactif Streamlit pour le suivi en direct (latence, précision glissante, dérives, utilisation CPU/RAM).

## 🧠 Modèles Incrémentaux Intégrés

1. **Hoeffding Tree (HT)** : Excellente interprétabilité.
2. **K-Nearest Neighbors (KNN)** : Capte les structures locales, excellente précision initiale.
3. **SGD / Régression Logistique** : Latence minimale, progression constante au fil du flux.

## 📁 Architecture du Projet

```text
automl_adaptatif/
├── data/
│   └── UNSW_NB15_training-set.csv     # Dataset
├── src/
│   ├── automl_engine.py               # Moteur principal
│   ├── models.py                      # Wrappers des modèles (HT, KNN, SGD)
│   ├── bandit.py                      # Algorithme ε-greedy
│   ├── drift.py                       # Détecteurs ADWIN & Page-Hinkley
│   ├── stream.py                      # Générateur de flux de données
│   └── monitoring.py                  # Thread de monitoring CPU/RAM
├── api/
│   └── main.py                        # Application FastAPI
├── dashboard/
│   └── app.py                         # Application Streamlit
├── results/                           # Historiques et métriques
├── tests/                             # Tests unitaires
└── README.md
```

## 🛠️ Prérequis et Installation

Assurez-vous d'avoir Python 3.9+ installé.

```bash
# Cloner le dépôt
git clone https://github.com/ReddaOutkida/Auto-ML-Adaptatif.git
cd Auto-ML-Adaptatif

# Installer les dépendances
pip install -r requirements.txt
```

## 🖥️ Lancement du Système

Le système est modulaire et nécessite de lancer trois terminaux séparés pour le moteur, l'API et le Dashboard.

**Terminal 1 — Moteur Auto-ML**
```bash
python -m src.automl_engine
```

**Terminal 2 — API FastAPI**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 3 — Dashboard Streamlit**
```bash
streamlit run dashboard/app.py
```

## 📊 Résultats
Sur une évaluation de 5 000 instances :
- Le modèle **KNN** s'est imposé comme modèle dominant avec une précision glissante de **87,66 %**.
- Détection d'une dérive de concept vers l'instance ~1 000, gérée avec succès par une réinitialisation automatique des scores du bandit et une ré-exploration.

## 👥 Auteurs
- **Ettaoussi Nouhaila**
- **Outkida Redda**
- **Ouyhia Ayoub**
- **Laiouej Anass**

*Projet encadré par : Pr. TABBAA Hiba*
