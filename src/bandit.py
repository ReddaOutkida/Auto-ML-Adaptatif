"""
bandit.py — Sélection dynamique de modèle par algorithme multi-armed bandit.

Ce module implémente deux stratégies de bandit :
- EpsilonGreedyBandit : exploration ε-greedy (exploration aléatoire vs exploitation).
- UCB1Bandit : Upper Confidence Bound pour un meilleur équilibre exploration/exploitation.

Ces stratégies permettent au moteur Auto-ML de sélectionner dynamiquement
le modèle le plus performant tout en continuant à explorer les alternatives.
"""

import logging
import math
import random
from datetime import datetime
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class EpsilonGreedyBandit:
    """
    Bandit epsilon-greedy pour la sélection de modèle.

    Avec probabilité epsilon, sélectionne un modèle au hasard (exploration).
    Sinon, sélectionne le modèle avec le meilleur score moyen (exploitation).
    """

    def __init__(self, model_names: List[str], epsilon: float = 0.1):
        """
        Args:
            model_names: Liste des noms de modèles candidats.
            epsilon: Probabilité d'exploration (0.0 à 1.0).
        """
        self.model_names = model_names
        self.epsilon = epsilon

        # Initialisation optimiste à 0.5
        self.scores: Dict[str, float] = {name: 0.5 for name in model_names}
        self.counts: Dict[str, int] = {name: 0 for name in model_names}
        self.current_model: str = model_names[0]

        # Historique : liste de tuples (timestamp, modèle_choisi, accuracy)
        self.history: List[tuple] = []

        logger.info(
            f"EpsilonGreedyBandit initialisé : modèles={model_names}, "
            f"epsilon={epsilon}"
        )

    def select_model(self) -> str:
        """
        Sélectionne le modèle à utiliser pour la prochaine prédiction.

        Returns:
            Nom du modèle sélectionné.
        """
        if random.random() < self.epsilon:
            # Exploration : modèle aléatoire
            choix = random.choice(self.model_names)
            logger.debug(f"Bandit — Exploration : modèle '{choix}' sélectionné")
        else:
            # Exploitation : meilleur score
            choix = max(self.scores, key=lambda m: self.scores[m])
            logger.debug(f"Bandit — Exploitation : modèle '{choix}' sélectionné")

        self.current_model = choix
        return choix

    def update(self, model_name: str, reward: float) -> None:
        """
        Met à jour le score moyen du modèle sélectionné.

        Utilise une moyenne incrémentale (running average).

        Args:
            model_name: Nom du modèle à mettre à jour.
            reward: Récompense reçue (1.0 = correct, 0.0 = erreur).
        """
        if model_name not in self.scores:
            logger.warning(f"Modèle inconnu : '{model_name}'")
            return

        self.counts[model_name] += 1
        n = self.counts[model_name]

        # Mise à jour de la moyenne glissante
        self.scores[model_name] += (reward - self.scores[model_name]) / n

        # Enregistrement dans l'historique
        self.history.append((
            datetime.now().isoformat(),
            model_name,
            self.scores[model_name]
        ))

    def reset_scores(self) -> None:
        """
        Remet tous les scores à 0.5 (initialisation optimiste).

        Utilisé après détection d'une dérive de concept pour repartir
        de zéro dans la sélection de modèle.
        """
        for name in self.model_names:
            self.scores[name] = 0.5
            self.counts[name] = 0
        logger.info("Bandit — Scores réinitialisés à 0.5 (dérive détectée)")

    def get_best_model(self) -> str:
        """
        Retourne le nom du modèle avec le meilleur score actuel.

        Returns:
            Nom du meilleur modèle.
        """
        return max(self.scores, key=lambda m: self.scores[m])

    def get_history_df(self) -> pd.DataFrame:
        """
        Retourne l'historique des sélections sous forme de DataFrame.

        Returns:
            DataFrame avec colonnes : timestamp, modèle, score.
        """
        if not self.history:
            return pd.DataFrame(columns=["timestamp", "modele", "score"])
        return pd.DataFrame(self.history, columns=["timestamp", "modele", "score"])

    def get_scores_summary(self) -> Dict[str, float]:
        """Retourne le résumé des scores actuels par modèle."""
        return dict(self.scores)

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyBandit(epsilon={self.epsilon}, "
            f"best={self.get_best_model()}, "
            f"scores={self.scores})"
        )


class UCB1Bandit:
    """
    Bandit UCB1 (Upper Confidence Bound) pour la sélection de modèle.

    Score UCB = moyenne + sqrt(2 * ln(total_count) / count_modèle)

    Favorise l'exploration des modèles peu testés tout en exploitant
    les bons modèles, avec une garantie théorique de regret logarithmique.
    """

    def __init__(self, model_names: List[str]):
        """
        Args:
            model_names: Liste des noms de modèles candidats.
        """
        self.model_names = model_names
        self.moyennes: Dict[str, float] = {name: 0.0 for name in model_names}
        self.counts: Dict[str, int] = {name: 0 for name in model_names}
        self.total_count: int = 0
        self.current_model: str = model_names[0]
        self.history: List[tuple] = []

        logger.info(f"UCB1Bandit initialisé : modèles={model_names}")

    def _ucb_score(self, model_name: str) -> float:
        """
        Calcule le score UCB pour un modèle.

        Args:
            model_name: Nom du modèle.

        Returns:
            Score UCB (infini si le modèle n'a jamais été essayé).
        """
        n = self.counts[model_name]
        if n == 0:
            return float("inf")  # Priorité aux modèles non testés
        return self.moyennes[model_name] + math.sqrt(
            2 * math.log(self.total_count) / n
        )

    def select_model(self) -> str:
        """
        Sélectionne le modèle avec le score UCB le plus élevé.

        Returns:
            Nom du modèle sélectionné.
        """
        choix = max(self.model_names, key=self._ucb_score)
        self.current_model = choix
        logger.debug(f"UCB1 — Modèle '{choix}' sélectionné")
        return choix

    def update(self, model_name: str, reward: float) -> None:
        """
        Met à jour la moyenne du modèle sélectionné.

        Args:
            model_name: Nom du modèle.
            reward: Récompense reçue (0.0 à 1.0).
        """
        if model_name not in self.moyennes:
            return

        self.total_count += 1
        self.counts[model_name] += 1
        n = self.counts[model_name]

        # Mise à jour de la moyenne incrémentale
        self.moyennes[model_name] += (reward - self.moyennes[model_name]) / n

        self.history.append((
            datetime.now().isoformat(),
            model_name,
            self.moyennes[model_name]
        ))

    def reset_scores(self) -> None:
        """Réinitialise toutes les statistiques."""
        for name in self.model_names:
            self.moyennes[name] = 0.0
            self.counts[name] = 0
        self.total_count = 0
        logger.info("UCB1Bandit — Scores réinitialisés")

    def get_best_model(self) -> str:
        """Retourne le modèle avec la meilleure moyenne actuelle."""
        return max(self.moyennes, key=lambda m: self.moyennes[m])

    def get_history_df(self) -> pd.DataFrame:
        """Retourne l'historique sous forme de DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=["timestamp", "modele", "score"])
        return pd.DataFrame(self.history, columns=["timestamp", "modele", "score"])

    def get_scores_summary(self) -> Dict[str, float]:
        """Retourne le résumé des scores UCB actuels."""
        return {
            name: self._ucb_score(name)
            for name in self.model_names
        }

    def __repr__(self) -> str:
        return (
            f"UCB1Bandit(best={self.get_best_model()}, "
            f"moyennes={self.moyennes})"
        )
