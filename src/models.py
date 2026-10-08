"""
models.py — Définition des trois modèles incrémentaux candidats.

Ce module encapsule les modèles river (Hoeffding Tree, KNN, SGD)
dans une classe ModelWrapper uniforme, permettant au moteur Auto-ML
de les utiliser de façon interchangeable.
"""

import logging
import time
from typing import Dict, Optional

from river import tree, neighbors, linear_model, optim, metrics, preprocessing

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Encapsule un modèle river avec une interface uniforme.

    Fournit des méthodes d'apprentissage, de prédiction et de métriques
    glissantes pour le moteur Auto-ML.
    """

    def __init__(self, name: str, model):
        """
        Args:
            name: Nom identifiant le modèle (ex: 'HT', 'KNN', 'SGD').
            model: Instance de modèle river compatible.
        """
        self.name = name
        self._model = model
        self._accuracy = metrics.Accuracy()
        self._kappa = metrics.CohenKappa()
        self.prediction_times: list = []  # latences en ms
        logger.info(f"Modèle '{name}' initialisé")

    def learn_one(self, x: dict, y: int) -> None:
        """
        Entraîne le modèle sur une seule instance.

        Args:
            x: Dictionnaire feature→valeur.
            y: Label vrai (0 ou 1).
        """
        self._model.learn_one(x, y)

    def predict_one(self, x: dict) -> int:
        """
        Prédit le label pour une instance, en mesurant la latence.

        Args:
            x: Dictionnaire feature→valeur.

        Returns:
            Label prédit (0 ou 1).
        """
        debut = time.perf_counter()
        prediction = self._model.predict_one(x)
        fin = time.perf_counter()

        latence_ms = (fin - debut) * 1000
        self.prediction_times.append(latence_ms)

        # Retourner 0 si le modèle n'a pas encore assez de données
        return int(prediction) if prediction is not None else 0

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        """
        Retourne les probabilités de classe pour une instance.

        Args:
            x: Dictionnaire feature→valeur.

        Returns:
            Dictionnaire {classe: probabilité}.
        """
        proba = self._model.predict_proba_one(x)
        if proba is None:
            return {0: 0.5, 1: 0.5}
        return proba

    def update_metrics(self, y_true: int, y_pred: int) -> None:
        """
        Met à jour les métriques glissantes avec un nouveau résultat.

        Args:
            y_true: Label vrai.
            y_pred: Label prédit.
        """
        self._accuracy.update(y_true, y_pred)
        self._kappa.update(y_true, y_pred)

    def get_accuracy(self) -> float:
        """
        Retourne la précision glissante courante.

        Returns:
            Précision (0.0 à 1.0). Retourne 0.5 si aucune prédiction.
        """
        val = self._accuracy.get()
        return val if val is not None and not (isinstance(val, float) and val != val) else 0.5

    def get_kappa(self) -> float:
        """
        Retourne la statistique de Cohen's Kappa courante.

        Returns:
            Kappa (-1.0 à 1.0). Retourne 0.0 si aucune prédiction.
        """
        val = self._kappa.get()
        return val if val is not None and not (isinstance(val, float) and val != val) else 0.0

    def get_avg_latency(self) -> float:
        """
        Retourne la latence moyenne de prédiction en millisecondes.

        Returns:
            Latence moyenne en ms.
        """
        if not self.prediction_times:
            return 0.0
        return sum(self.prediction_times) / len(self.prediction_times)

    def reset(self) -> None:
        """
        Réinitialise le modèle et ses métriques.
        Utilisé après détection d'une dérive de concept.
        """
        # Recréer le modèle selon son type
        if isinstance(self._model, tree.HoeffdingTreeClassifier):
            self._model = tree.HoeffdingTreeClassifier(
                grace_period=100,
                delta=1e-5,
                split_criterion="gini"
            )
        elif isinstance(self._model, neighbors.KNNClassifier):
            self._model = neighbors.KNNClassifier(n_neighbors=5)
        elif isinstance(self._model, linear_model.LogisticRegression):
            self._model = linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=0.01)
            )

        self._accuracy = metrics.Accuracy()
        self._kappa = metrics.CohenKappa()
        self.prediction_times = []
        logger.info(f"Modèle '{self.name}' réinitialisé après dérive")

    def __repr__(self) -> str:
        return (
            f"ModelWrapper(name={self.name!r}, "
            f"accuracy={self.get_accuracy():.4f}, "
            f"kappa={self.get_kappa():.4f})"
        )


def _creer_hoeffding_tree() -> ModelWrapper:
    """Crée un Hoeffding Tree Classifier."""
    modele = tree.HoeffdingTreeClassifier(
        grace_period=100,
        delta=1e-5,
        split_criterion="gini"
    )
    return ModelWrapper("HT", modele)


def _creer_knn() -> ModelWrapper:
    """Crée un KNN Classifier incrémental."""
    modele = neighbors.KNNClassifier(n_neighbors=5)
    return ModelWrapper("KNN", modele)


def _creer_sgd() -> ModelWrapper:
    """Crée un modèle de régression logistique avec SGD."""
    modele = linear_model.LogisticRegression(
        optimizer=optim.SGD(lr=0.01)
    )
    return ModelWrapper("SGD", modele)


# Dictionnaire des modèles candidats instanciés
CANDIDATE_MODELS: Dict[str, ModelWrapper] = {
    "HT": _creer_hoeffding_tree(),
    "KNN": _creer_knn(),
    "SGD": _creer_sgd(),
}

logger.info(f"Modèles candidats initialisés : {list(CANDIDATE_MODELS.keys())}")
