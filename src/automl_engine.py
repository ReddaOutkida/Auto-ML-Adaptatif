"""
automl_engine.py — Moteur principal du système Auto-ML Adaptatif.

Ce module orchestre l'ensemble du pipeline : flux de données,
sélection dynamique de modèle via bandit multi-bras, détection
de dérive de concept, et logging des métriques de performance.

Utilisation :
    python -m src.automl_engine
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stream import stream_dataset
from src.models import CANDIDATE_MODELS, ModelWrapper
from src.bandit import EpsilonGreedyBandit
from src.drift import DriftDetector
from src.monitoring import SystemMonitor

logger = logging.getLogger(__name__)

# Répertoire de sortie pour les résultats
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class AutoMLEngine:
    """
    Moteur Auto-ML adaptatif avec sélection dynamique de modèle.

    Orchestre le flux de données, le bandit multi-bras pour la
    sélection de modèle, et la détection de dérive de concept
    pour maintenir une précision optimale sans intervention humaine.
    """

    def __init__(
        self,
        filepath: str,
        epsilon: float = 0.1,
        window_size: int = 10000,
        eval_every: int = 500,
        drift_delta: float = 0.002,
    ):
        """
        Args:
            filepath: Chemin vers le dataset CSV UNSW-NB15.
            epsilon: Paramètre d'exploration du bandit epsilon-greedy.
            window_size: Taille de la fenêtre glissante (non utilisé directement,
                         river gère ça en interne).
            eval_every: Fréquence de logging des métriques (en instances).
            drift_delta: Sensibilité du détecteur ADWIN.
        """
        self.filepath = filepath
        self.epsilon = epsilon
        self.eval_every = eval_every

        # Copie des modèles candidats (éviter la mutation du singleton)
        from river import tree, neighbors, linear_model, optim
        from src.models import ModelWrapper

        self.modeles: Dict[str, ModelWrapper] = {
            "HT": ModelWrapper("HT", tree.HoeffdingTreeClassifier(
                grace_period=100, delta=1e-5, split_criterion="gini"
            )),
            "KNN": ModelWrapper("KNN", neighbors.KNNClassifier(n_neighbors=5)),
            "SGD": ModelWrapper("SGD", linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=0.01)
            )),
        }

        # Bandit et détecteur de dérive
        self.bandit = EpsilonGreedyBandit(
            model_names=list(self.modeles.keys()),
            epsilon=epsilon
        )
        self.detecteur_derive = DriftDetector(adwin_delta=drift_delta)

        # Monitoring système
        self.moniteur = SystemMonitor(interval=1.0)

        # État interne
        self._instance_count: int = 0
        self._nb_drifts: int = 0
        self._historique_metriques: List[Dict] = []
        self._historique_changements: List[Dict] = []
        self._modele_actif: str = list(self.modeles.keys())[0]
        self._debut: float = time.time()
        self._en_cours: bool = False

        logger.info(
            f"AutoMLEngine initialisé — fichier={filepath}, "
            f"epsilon={epsilon}, eval_every={eval_every}"
        )

    def run(self, max_instances: int = 50000) -> None:
        """
        Lance la boucle principale d'apprentissage en ligne.

        Pour chaque instance du flux :
        1. Le bandit sélectionne le modèle actif.
        2. Le modèle actif prédit.
        3. L'erreur est calculée et passée au détecteur de dérive.
        4. Tous les modèles apprennent sur l'instance (apprentissage passif).
        5. Le bandit est mis à jour avec la récompense.
        6. Toutes les eval_every instances, les métriques sont loggées.
        7. Si une dérive est détectée, le bandit est réinitialisé.

        Args:
            max_instances: Nombre maximum d'instances à traiter.
        """
        self._en_cours = True
        self.moniteur.start()

        logger.info(
            f"Démarrage de la boucle Auto-ML — max_instances={max_instances}"
        )

        try:
            flux = stream_dataset(self.filepath, shuffle=True)

            for x, y in flux:
                if self._instance_count >= max_instances:
                    break

                self._instance_count += 1

                # 1. Sélection du modèle actif par le bandit
                nom_modele = self.bandit.select_model()

                ancien_modele = self._modele_actif
                self._modele_actif = nom_modele

                # Enregistrer les changements de modèle
                if ancien_modele != nom_modele:
                    self._historique_changements.append({
                        "timestamp": datetime.now().isoformat(),
                        "instance": self._instance_count,
                        "ancien_modele": ancien_modele,
                        "nouveau_modele": nom_modele,
                        "raison": "bandit_selection",
                    })

                # 2. Prédictions PRE-learning pour TOUS les modèles
                # (évaluation avant apprentissage = mesure correcte des erreurs)
                preds = {nom: modele.predict_one(x) for nom, modele in self.modeles.items()}
                y_pred = preds[nom_modele]

                # 3. Calcul de l'erreur du modèle actif
                erreur = int(y_pred != y)

                # 4. Mise à jour du détecteur de dérive
                self.detecteur_derive.update(float(erreur))

                # 5. Apprentissage passif — tous les modèles apprennent
                # Les métriques utilisent les prédictions PRE-learning (step 2)
                for nom, modele in self.modeles.items():
                    modele.learn_one(x, y)
                    modele.update_metrics(y, preds[nom])

                # 6. Mise à jour du bandit avec reward binaire pour TOUS les modèles.
                # Chaque modèle reçoit 1.0 s'il a bien prédit, 0.0 sinon.
                # La moyenne glissante converge vers la vraie accuracy de chaque modèle,
                # ce qui permet au bandit de basculer rapidement vers le meilleur.
                for nom_b in self.modeles:
                    recompense_b = 1.0 - int(preds[nom_b] != y)
                    self.bandit.update(nom_b, recompense_b)

                # 7. Logging périodique des métriques
                if self._instance_count % self.eval_every == 0:
                    self._logger_metriques()

                # 8. Gestion de la dérive détectée
                if self.detecteur_derive.drift_detected():
                    self._nb_drifts += 1
                    detecteur = self.detecteur_derive.get_detector_name()

                    logger.warning(
                        f"[AUTO-ML] Dérive #{self._nb_drifts} à l'instance "
                        f"{self._instance_count} — Détecteur: {detecteur} — "
                        f"Reset du bandit"
                    )

                    # Reset du bandit
                    self.bandit.reset_scores()

                    # Enregistrer l'événement de dérive
                    self._historique_changements.append({
                        "timestamp": datetime.now().isoformat(),
                        "instance": self._instance_count,
                        "ancien_modele": nom_modele,
                        "nouveau_modele": nom_modele,
                        "raison": f"derive_{detecteur}",
                    })

                    # Réinitialiser le détecteur
                    self.detecteur_derive.reset()

        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        except StopIteration:
            logger.info("Flux de données épuisé")
        finally:
            self._en_cours = False
            self.moniteur.stop()
            self._sauvegarder_historique()

        logger.info(
            f"Boucle Auto-ML terminée — {self._instance_count} instances traitées, "
            f"{self._nb_drifts} dérives détectées"
        )

    def _logger_metriques(self) -> None:
        """Loggue et enregistre les métriques courantes."""
        metriques_sys = self.moniteur.get_current()
        entree = {
            "instance": self._instance_count,
            "timestamp": datetime.now().isoformat(),
            "modele_actif": self._modele_actif,
            "nb_drifts": self._nb_drifts,
        }

        for nom, modele in self.modeles.items():
            entree[f"accuracy_{nom}"] = modele.get_accuracy()
            entree[f"kappa_{nom}"] = modele.get_kappa()
            entree[f"latency_ms_{nom}"] = modele.get_avg_latency()
            entree[f"score_bandit_{nom}"] = round(self.bandit.scores.get(nom, 0.5), 4)

        entree["cpu_percent"] = metriques_sys["cpu_percent"]
        entree["ram_percent"] = metriques_sys["ram_percent"]

        self._historique_metriques.append(entree)

        # Sauvegarde incrémentale dans results/history.csv à chaque checkpoint
        # (permet à l'API de lire les données pendant l'exécution)
        self._sauvegarder_historique()

        # Affichage dans les logs
        accs = {n: f"{m.get_accuracy():.4f}" for n, m in self.modeles.items()}
        logger.info(
            f"Instance {self._instance_count:6d} | "
            f"Modèle actif: {self._modele_actif:3s} | "
            f"Précisions: {accs} | "
            f"Dérives: {self._nb_drifts} | "
            f"CPU: {metriques_sys['cpu_percent']:.1f}% | "
            f"RAM: {metriques_sys['ram_percent']:.1f}%"
        )

    def _sauvegarder_historique(self) -> None:
        """Sauvegarde l'historique complet dans results/history.csv."""
        if self._historique_metriques:
            df = pd.DataFrame(self._historique_metriques)
            chemin = RESULTS_DIR / "history.csv"
            df.to_csv(chemin, index=False)
            logger.info(f"Historique sauvegardé : {chemin}")

        if self._historique_changements:
            df_ch = pd.DataFrame(self._historique_changements)
            chemin_ch = RESULTS_DIR / "model_changes.csv"
            df_ch.to_csv(chemin_ch, index=False)
            logger.info(f"Changements de modèle sauvegardés : {chemin_ch}")

    def get_live_metrics(self) -> Dict:
        """
        Retourne les métriques actuelles sous forme JSON-serializable.

        Returns:
            Dictionnaire avec précisions, modèle actif, dérives, latence,
            CPU et RAM.
        """
        metriques_sys = self.moniteur.get_current()
        uptime = time.time() - self._debut

        return {
            "modele_actif": self._modele_actif,
            "instance_count": self._instance_count,
            "nb_drifts": self._nb_drifts,
            "uptime_s": round(uptime, 1),
            "en_cours": self._en_cours,
            "modeles": {
                nom: {
                    "accuracy": round(modele.get_accuracy(), 4),
                    "kappa": round(modele.get_kappa(), 4),
                    "latency_ms": round(modele.get_avg_latency(), 3),
                    "score_bandit": round(self.bandit.scores.get(nom, 0.5), 4),
                }
                for nom, modele in self.modeles.items()
            },
            "systeme": {
                "cpu_percent": metriques_sys["cpu_percent"],
                "ram_percent": metriques_sys["ram_percent"],
                "ram_utilisee_mb": round(metriques_sys["ram_utilisee_mb"], 1),
            },
        }

    def get_full_history(self) -> pd.DataFrame:
        """
        Retourne tout l'historique des métriques.

        Returns:
            DataFrame avec l'historique complet.
        """
        if not self._historique_metriques:
            return pd.DataFrame()
        return pd.DataFrame(self._historique_metriques)

    def get_model_changes(self) -> pd.DataFrame:
        """Retourne l'historique des changements de modèle."""
        if not self._historique_changements:
            return pd.DataFrame()
        return pd.DataFrame(self._historique_changements)


# Point d'entrée principal
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    parser = argparse.ArgumentParser(description="Moteur Auto-ML Adaptatif")
    parser.add_argument(
        "--data",
        default="data/UNSW_NB15_training-set.csv",
        help="Chemin vers le dataset CSV"
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=50000,
        help="Nombre maximum d'instances à traiter"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon pour le bandit epsilon-greedy"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Fréquence de logging des métriques"
    )
    args = parser.parse_args()

    # Résoudre le chemin par rapport à la racine du projet
    racine = Path(__file__).parent.parent
    chemin_data = racine / args.data

    moteur = AutoMLEngine(
        filepath=str(chemin_data),
        epsilon=args.epsilon,
        eval_every=args.eval_every,
    )

    moteur.run(max_instances=args.max_instances)

    # Afficher un résumé final
    metriques = moteur.get_live_metrics()
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"Instances traitées : {metriques['instance_count']}")
    print(f"Dérives détectées  : {metriques['nb_drifts']}")
    print(f"Meilleur modèle    : {moteur.bandit.get_best_model()}")
    for nom, stats in metriques["modeles"].items():
        print(
            f"  {nom:4s} | Accuracy={stats['accuracy']:.4f} | "
            f"Kappa={stats['kappa']:.4f} | "
            f"Latence={stats['latency_ms']:.2f}ms"
        )
    print("=" * 60)
