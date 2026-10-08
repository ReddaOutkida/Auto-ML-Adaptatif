"""
stream.py — Générateur de flux de données pour le dataset UNSW-NB15.

Ce module charge le dataset CSV et le transforme en flux de données
incrémental (streaming), simulant un environnement temps réel.
Il supporte aussi l'injection de dérives artificielles pour tester
la robustesse du système de détection.
"""

import logging
import random
from pathlib import Path
from typing import Generator, Tuple, Dict

import numpy as np
import pandas as pd
from river import preprocessing

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Colonnes non numériques à supprimer
COLONNES_A_SUPPRIMER = ["proto", "service", "state", "attack_cat"]
COLONNE_LABEL = "label"


def _charger_dataframe(filepath: str) -> pd.DataFrame:
    """Charge et nettoie le dataset UNSW-NB15."""
    chemin = Path(filepath)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {filepath}\n"
            "Téléchargez UNSW_NB15_training-set.csv depuis Kaggle :\n"
            "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
        )

    df = pd.read_csv(filepath)
    logger.info(f"Dataset chargé : {df.shape[0]} instances, {df.shape[1]} colonnes")

    # Supprimer les colonnes non numériques et inutiles
    colonnes_a_drop = [c for c in COLONNES_A_SUPPRIMER if c in df.columns]
    df = df.drop(columns=colonnes_a_drop, errors="ignore")

    # Vérifier la présence du label
    if COLONNE_LABEL not in df.columns:
        raise ValueError(f"Colonne '{COLONNE_LABEL}' introuvable dans le dataset.")

    # Conserver uniquement les colonnes numériques + label
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
    if COLONNE_LABEL not in colonnes_numeriques:
        colonnes_numeriques.append(COLONNE_LABEL)
    df = df[colonnes_numeriques]

    # Supprimer les valeurs manquantes
    df = df.dropna()

    # Log de la distribution des classes
    dist = df[COLONNE_LABEL].value_counts().to_dict()
    logger.info(f"Distribution des classes : {dist}")
    logger.info(f"Features conservées : {[c for c in df.columns if c != COLONNE_LABEL]}")

    return df


def stream_dataset(
    filepath: str, shuffle: bool = True
) -> Generator[Tuple[Dict[str, float], int], None, None]:
    """
    Générateur de flux incrémental depuis le dataset UNSW-NB15.

    Yield des tuples (x_dict, y) où x_dict est un dictionnaire
    feature→valeur normalisé et y est le label binaire (0 ou 1).

    Args:
        filepath: Chemin vers le fichier CSV UNSW-NB15.
        shuffle: Si True, mélange le dataset avant de le streamer.

    Yields:
        Tuple (dict de features normalisées, label int)
    """
    df = _charger_dataframe(filepath)

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Dataset mélangé (shuffle=True)")

    # Scaler incrémental river pour normalisation en ligne
    scaler = preprocessing.StandardScaler()

    features = [c for c in df.columns if c != COLONNE_LABEL]
    total = len(df)
    logger.info(f"Démarrage du flux : {total} instances à streamer")

    for i, row in enumerate(df.itertuples(index=False)):
        x = {feat: float(getattr(row, feat)) for feat in features}
        y = int(getattr(row, COLONNE_LABEL))

        # Mise à jour et transformation du scaler incrémental
        x_scaled = dict(scaler.transform_one(x))
        scaler.learn_one(x)

        yield x_scaled, y

    logger.info(f"Flux terminé : {total} instances streamées")


def stream_with_artificial_drift(
    filepath: str, drift_at: int = 5000
) -> Generator[Tuple[Dict[str, float], int], None, None]:
    """
    Flux avec dérive artificielle introduite à l'instance drift_at.

    Avant drift_at : flux normal.
    Après drift_at : certaines features sont permutées aléatoirement
    pour simuler un changement de distribution.

    Args:
        filepath: Chemin vers le fichier CSV UNSW-NB15.
        drift_at: Numéro d'instance où la dérive est introduite.

    Yields:
        Tuple (dict de features, label int)
    """
    df = _charger_dataframe(filepath)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    scaler = preprocessing.StandardScaler()
    features = [c for c in df.columns if c != COLONNE_LABEL]

    rng = random.Random(99)
    drift_logged = False

    for i, row in enumerate(df.itertuples(index=False)):
        x = {feat: float(getattr(row, feat)) for feat in features}
        y = int(getattr(row, COLONNE_LABEL))

        x_scaled = dict(scaler.transform_one(x))
        scaler.learn_one(x)

        # Injection de la dérive artificielle à partir de drift_at
        if i >= drift_at:
            if not drift_logged:
                logger.warning(
                    f"[DÉRIVE ARTIFICIELLE] Introduite à l'instance {i} "
                    f"— permutation des features activée"
                )
                drift_logged = True

            # Permuter aléatoirement les valeurs de certaines features
            vals = list(x_scaled.values())
            rng.shuffle(vals)
            x_scaled = dict(zip(x_scaled.keys(), vals))

        yield x_scaled, y

    logger.info("Flux avec dérive artificielle terminé")
