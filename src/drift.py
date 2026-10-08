"""
drift.py — Détection de dérive de concept (Concept Drift Detection).

Ce module implémente un détecteur de dérive combiné utilisant
ADWIN et Page-Hinkley en parallèle pour une détection robuste
des changements de distribution dans le flux de données.
"""

import logging
from datetime import datetime
from typing import Optional

from river.drift import ADWIN, PageHinkley

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Détecteur de dérive combiné (ADWIN + Page-Hinkley).

    Fait tourner deux détecteurs en parallèle et signale une dérive
    dès que l'un d'eux en détecte une. Loggue chaque détection
    avec le timestamp, le numéro d'instance et le nom du détecteur.
    """

    def __init__(self, adwin_delta: float = 0.002, ph_delta: float = 0.005):
        """
        Args:
            adwin_delta: Paramètre delta pour ADWIN (sensibilité).
                         Plus petit = plus sensible.
            ph_delta: Paramètre delta pour Page-Hinkley.
        """
        self._adwin = ADWIN(delta=adwin_delta)
        self._page_hinkley = PageHinkley(min_instances=30, delta=ph_delta)

        self._detecteur_ayant_detecte: Optional[str] = None
        self._nb_drifts = 0
        self._instance_courante = 0

        logger.info(
            f"DriftDetector initialisé : ADWIN(delta={adwin_delta}), "
            f"PageHinkley(delta={ph_delta})"
        )

    def update(self, error: float) -> None:
        """
        Met à jour les deux détecteurs avec l'erreur de prédiction.

        Args:
            error: Erreur de prédiction (0.0 = correct, 1.0 = erreur).
        """
        self._instance_courante += 1
        self._detecteur_ayant_detecte = None

        self._adwin.update(error)
        self._page_hinkley.update(error)

        # Vérifier quel détecteur a signalé une dérive
        if self._adwin.drift_detected and self._page_hinkley.drift_detected:
            self._detecteur_ayant_detecte = "ADWIN+PageHinkley"
        elif self._adwin.drift_detected:
            self._detecteur_ayant_detecte = "ADWIN"
        elif self._page_hinkley.drift_detected:
            self._detecteur_ayant_detecte = "PageHinkley"

        # Logger la dérive si détectée
        if self._detecteur_ayant_detecte is not None:
            self._nb_drifts += 1
            logger.warning(
                f"[DÉRIVE DÉTECTÉE #{self._nb_drifts}] "
                f"Timestamp: {datetime.now().isoformat()} | "
                f"Instance: {self._instance_courante} | "
                f"Détecteur: {self._detecteur_ayant_detecte}"
            )

    def drift_detected(self) -> bool:
        """
        Vérifie si une dérive a été détectée lors du dernier update.

        Returns:
            True si au moins un détecteur signale une dérive.
        """
        return self._detecteur_ayant_detecte is not None

    def get_detector_name(self) -> str:
        """
        Retourne le nom du détecteur ayant signalé la dérive.

        Returns:
            Nom du détecteur ('ADWIN', 'PageHinkley', 'ADWIN+PageHinkley')
            ou 'Aucun' si pas de dérive.
        """
        return self._detecteur_ayant_detecte or "Aucun"

    def get_nb_drifts(self) -> int:
        """Retourne le nombre total de dérives détectées depuis le début."""
        return self._nb_drifts

    def get_instance_count(self) -> int:
        """Retourne le nombre d'instances traitées."""
        return self._instance_courante

    def reset(self) -> None:
        """
        Réinitialise les deux détecteurs.

        À appeler après chaque détection de dérive pour repartir
        sur une base propre.
        """
        self._adwin = ADWIN(delta=self._adwin.delta)
        self._page_hinkley = PageHinkley(
            min_instances=30,
            delta=self._page_hinkley.delta
        )
        self._detecteur_ayant_detecte = None
        logger.info(
            f"DriftDetector réinitialisé à l'instance {self._instance_courante}"
        )
