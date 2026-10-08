"""
monitoring.py — Surveillance des ressources système (CPU, RAM, latence).

Ce module surveille en continu les ressources système dans un thread
daemon, permettant au dashboard de les afficher en temps réel sans
bloquer le moteur Auto-ML.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import psutil

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Moniteur de ressources système en arrière-plan.

    Collecte périodiquement CPU%, RAM% et RAM utilisée (MB)
    dans un thread daemon séparé.
    """

    def __init__(self, interval: float = 1.0):
        """
        Args:
            interval: Intervalle de mesure en secondes.
        """
        self.interval = interval
        self._historique: List[Dict] = []
        self._thread: threading.Thread = None
        self._actif: bool = False
        self._lock = threading.Lock()

        logger.info(f"SystemMonitor initialisé (intervalle={interval}s)")

    def start(self) -> None:
        """
        Lance la mesure des ressources en arrière-plan (thread daemon).

        Le thread s'arrête automatiquement quand le programme principal
        se termine (daemon=True).
        """
        if self._actif:
            logger.warning("SystemMonitor déjà en cours d'exécution")
            return

        self._actif = True
        self._thread = threading.Thread(
            target=self._boucle_mesure,
            daemon=True,
            name="SystemMonitor"
        )
        self._thread.start()
        logger.info("SystemMonitor démarré")

    def stop(self) -> None:
        """Arrête la mesure des ressources."""
        self._actif = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("SystemMonitor arrêté")

    def _boucle_mesure(self) -> None:
        """Boucle interne de mesure (exécutée dans le thread daemon)."""
        while self._actif:
            mesure = self._mesurer()
            with self._lock:
                self._historique.append(mesure)
                # Conserver uniquement les 3600 dernières mesures (1h à 1s/mesure)
                if len(self._historique) > 3600:
                    self._historique = self._historique[-3600:]
            time.sleep(self.interval)

    def _mesurer(self) -> Dict:
        """Effectue une mesure instantanée des ressources."""
        memoire = psutil.virtual_memory()
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": memoire.percent,
            "ram_utilisee_mb": memoire.used / (1024 * 1024),
            "ram_totale_mb": memoire.total / (1024 * 1024),
        }

    def get_current(self) -> Dict:
        """
        Retourne les métriques instantanées actuelles.

        Returns:
            Dictionnaire avec cpu_percent, ram_percent, ram_utilisee_mb.
        """
        return self._mesurer()

    def get_history(self) -> pd.DataFrame:
        """
        Retourne l'historique complet des mesures.

        Returns:
            DataFrame avec colonnes : timestamp, cpu_percent, ram_percent,
            ram_utilisee_mb, ram_totale_mb.
        """
        with self._lock:
            if not self._historique:
                return pd.DataFrame(columns=[
                    "timestamp", "cpu_percent", "ram_percent",
                    "ram_utilisee_mb", "ram_totale_mb"
                ])
            return pd.DataFrame(self._historique)

    def get_recent_history(self, secondes: int = 60) -> pd.DataFrame:
        """
        Retourne l'historique des N dernières secondes.

        Args:
            secondes: Nombre de secondes à conserver.

        Returns:
            DataFrame avec les mesures récentes.
        """
        n = int(secondes / self.interval)
        with self._lock:
            recent = self._historique[-n:] if len(self._historique) >= n else self._historique
            if not recent:
                return pd.DataFrame(columns=[
                    "timestamp", "cpu_percent", "ram_percent",
                    "ram_utilisee_mb", "ram_totale_mb"
                ])
            return pd.DataFrame(recent)

    def __repr__(self) -> str:
        actif = "actif" if self._actif else "inactif"
        return f"SystemMonitor({actif}, {len(self._historique)} mesures enregistrées)"
