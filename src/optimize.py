"""
optimize.py — Optimisation des hyperparamètres via Optuna.

Utilise Optuna pour trouver les meilleurs hyperparamètres du moteur Auto-ML
en utilisant les 5000 premières instances du dataset comme ensemble de validation.

Paramètres optimisés :
- epsilon du bandit (0.01 à 0.3)
- grace_period du Hoeffding Tree (50 à 500)
- lr du SGD (0.001 à 0.1)
- delta d'ADWIN (0.0001 à 0.01)

Utilisation :
    python -m src.optimize --data data/UNSW_NB15_training-set.csv --n-trials 50
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stream import stream_dataset
from src.drift import DriftDetector
from src.bandit import EpsilonGreedyBandit
from src.models import ModelWrapper

logger = logging.getLogger(__name__)

# Réduire la verbosité d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_INSTANCES_OPTIM = 5000  # Instances utilisées pour l'optimisation


def _evaluer_configuration(
    filepath: str,
    epsilon: float,
    grace_period: int,
    lr: float,
    adwin_delta: float,
    n_instances: int = N_INSTANCES_OPTIM,
) -> float:
    """
    Évalue une configuration d'hyperparamètres sur N instances.

    Args:
        filepath: Chemin vers le dataset.
        epsilon: Taux d'exploration du bandit.
        grace_period: Période de grâce du Hoeffding Tree.
        lr: Taux d'apprentissage du SGD.
        adwin_delta: Sensibilité du détecteur ADWIN.
        n_instances: Nombre d'instances pour l'évaluation.

    Returns:
        Précision moyenne des 3 modèles après n_instances instances.
    """
    from river import tree, neighbors, linear_model, optim, metrics

    # Instancier les modèles avec les hyperparamètres à tester
    modeles = {
        "HT": ModelWrapper("HT", tree.HoeffdingTreeClassifier(
            grace_period=grace_period,
            delta=1e-5,
            split_criterion="gini"
        )),
        "KNN": ModelWrapper("KNN", neighbors.KNNClassifier(n_neighbors=5)),
        "SGD": ModelWrapper("SGD", linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=lr)
        )),
    }

    bandit = EpsilonGreedyBandit(
        model_names=list(modeles.keys()),
        epsilon=epsilon
    )
    detecteur = DriftDetector(adwin_delta=adwin_delta)

    compteur = 0

    try:
        flux = stream_dataset(filepath, shuffle=True)
        for x, y in flux:
            if compteur >= n_instances:
                break
            compteur += 1

            # Sélection + prédiction
            nom = bandit.select_model()
            y_pred = modeles[nom].predict_one(x)
            erreur = int(y_pred != y)

            # Mise à jour drift
            detecteur.update(float(erreur))

            # Apprentissage passif
            for m in modeles.values():
                m.learn_one(x, y)
                m.update_metrics(y, m.predict_one(x))

            # Mise à jour bandit
            bandit.update(nom, 1.0 - erreur)

            # Reset si dérive
            if detecteur.drift_detected():
                bandit.reset_scores()
                detecteur.reset()

    except FileNotFoundError:
        raise

    # Calculer la précision moyenne des 3 modèles
    precision_moyenne = sum(m.get_accuracy() for m in modeles.values()) / len(modeles)
    return precision_moyenne


def objectif(trial: optuna.Trial, filepath: str) -> float:
    """
    Fonction objectif pour Optuna.

    Args:
        trial: Essai Optuna courant.
        filepath: Chemin vers le dataset.

    Returns:
        Précision moyenne à maximiser.
    """
    # Définition de l'espace de recherche
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3)
    grace_period = trial.suggest_int("grace_period", 50, 500)
    lr = trial.suggest_float("lr", 0.001, 0.1, log=True)
    adwin_delta = trial.suggest_float("adwin_delta", 0.0001, 0.01, log=True)

    precision = _evaluer_configuration(
        filepath=filepath,
        epsilon=epsilon,
        grace_period=grace_period,
        lr=lr,
        adwin_delta=adwin_delta,
    )

    logger.info(
        f"Trial {trial.number}: epsilon={epsilon:.3f}, "
        f"grace_period={grace_period}, lr={lr:.4f}, "
        f"adwin_delta={adwin_delta:.5f} → précision={precision:.4f}"
    )

    return precision


def optimiser(
    filepath: str,
    n_trials: int = 50,
    timeout: int = 600,
) -> Dict:
    """
    Lance l'optimisation Optuna et retourne les meilleurs paramètres.

    Args:
        filepath: Chemin vers le dataset.
        n_trials: Nombre d'essais Optuna.
        timeout: Temps maximum en secondes.

    Returns:
        Dictionnaire des meilleurs hyperparamètres.
    """
    logger.info(
        f"Démarrage de l'optimisation Optuna — "
        f"{n_trials} essais, timeout={timeout}s"
    )

    etude = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="automl_adaptatif",
    )

    etude.optimize(
        lambda trial: objectif(trial, filepath),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    meilleurs = etude.best_params
    meilleure_precision = etude.best_value

    print("\n" + "=" * 60)
    print("MEILLEURS HYPERPARAMÈTRES TROUVÉS PAR OPTUNA")
    print("=" * 60)
    print(f"Précision moyenne obtenue : {meilleure_precision:.4f}")
    print(f"epsilon        : {meilleurs['epsilon']:.4f}")
    print(f"grace_period   : {meilleurs['grace_period']}")
    print(f"lr             : {meilleurs['lr']:.5f}")
    print(f"adwin_delta    : {meilleurs['adwin_delta']:.6f}")
    print("=" * 60)

    return meilleurs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Optimisation Optuna pour Auto-ML Adaptatif"
    )
    parser.add_argument(
        "--data",
        default="data/UNSW_NB15_training-set.csv",
        help="Chemin vers le dataset CSV"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Nombre d'essais Optuna"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Temps maximum en secondes"
    )
    args = parser.parse_args()

    racine = Path(__file__).parent.parent
    chemin_data = racine / args.data

    meilleurs_params = optimiser(
        filepath=str(chemin_data),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\nVous pouvez maintenant utiliser ces paramètres dans AutoMLEngine :")
    print(f"AutoMLEngine(filepath=..., epsilon={meilleurs_params['epsilon']:.4f}, ...)")
