"""
main.py — API FastAPI pour le moteur Auto-ML Adaptatif.

Expose des endpoints REST permettant d'interroger l'état du système,
de faire des prédictions, et de contrôler le moteur Auto-ML.

Lancement :
    uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ajouter la racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.automl_engine import AutoMLEngine

logger = logging.getLogger(__name__)

# ─── Initialisation de l'application ───────────────────────────────────────

app = FastAPI(
    title="Auto-ML Adaptatif API",
    description=(
        "API REST pour le système Auto-ML Adaptatif avec sélection dynamique "
        "de modèle et détection de dérive de concept."
    ),
    version="1.0.0",
)

# Autoriser toutes les origines pour le développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Horodatage de démarrage
_DEBUT = time.time()

# Instance globale du moteur Auto-ML (initialisée au démarrage)
_MOTEUR: AutoMLEngine = None


@app.on_event("startup")
async def demarrage():
    """Initialise le moteur Auto-ML au démarrage de l'API."""
    global _MOTEUR
    chemin_data = Path(__file__).parent.parent / "data" / "UNSW_NB15_training-set.csv"
    _MOTEUR = AutoMLEngine(
        filepath=str(chemin_data),
        epsilon=0.1,
        eval_every=500,
    )
    logger.info("Moteur Auto-ML initialisé (API prête — lancez le moteur séparément)")


def _verifier_moteur():
    """Vérifie que le moteur est initialisé."""
    if _MOTEUR is None:
        raise HTTPException(status_code=503, detail="Moteur Auto-ML non initialisé")


# ─── Modèles Pydantic ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Corps de la requête POST /predict."""
    features: Dict[str, float]


class PredictResponse(BaseModel):
    """Corps de la réponse POST /predict."""
    prediction: int
    model_used: str
    latency_ms: float
    probabilities: Dict[str, float]


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", summary="État de santé de l'API")
async def health() -> Dict[str, Any]:
    """
    Vérifie que l'API est opérationnelle.

    Returns:
        Statut et durée de fonctionnement en secondes.
    """
    return {
        "status": "ok",
        "uptime": round(time.time() - _DEBUT, 1),
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse, summary="Prédiction d'une instance")
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Prédit le label d'une instance avec le modèle actif.

    Body:
        features: Dictionnaire des features (noms → valeurs float).

    Returns:
        Prédiction (0/1), modèle utilisé, latence en ms.
    """
    _verifier_moteur()

    nom_modele = _MOTEUR.bandit.get_best_model()
    modele = _MOTEUR.modeles[nom_modele]

    import time as t
    debut = t.perf_counter()
    prediction = modele.predict_one(request.features)
    proba = modele.predict_proba_one(request.features)
    fin = t.perf_counter()

    latence_ms = (fin - debut) * 1000

    return PredictResponse(
        prediction=prediction,
        model_used=nom_modele,
        latency_ms=round(latence_ms, 3),
        probabilities={str(k): round(v, 4) for k, v in proba.items()},
    )


def _lire_derniere_ligne_csv() -> pd.Series:
    """Lit la dernière ligne de results/history.csv. Retourne une Series vide si absent."""
    csv_path = Path(__file__).parent.parent / "results" / "history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df.iloc[-1]
        except Exception:
            pass
    return pd.Series(dtype=object)


@app.get("/metrics", summary="Métriques courantes du système")
async def metrics() -> Dict[str, Any]:
    """
    Retourne toutes les métriques courantes.

    Si le moteur local n'a pas encore traité d'instances (processus séparé),
    lit les données depuis results/history.csv.

    Returns:
        Précisions par modèle, modèle actif, nombre de dérives,
        métriques système (CPU, RAM), latences.
    """
    _verifier_moteur()
    data = _MOTEUR.get_live_metrics()

    # Fallback CSV si le moteur local est vide (processus séparés)
    if data["instance_count"] == 0:
        last = _lire_derniere_ligne_csv()
        if not last.empty:
            data["instance_count"] = int(last.get("instance", 0))
            data["nb_drifts"] = int(last.get("nb_drifts", 0))
            data["modele_actif"] = str(last.get("modele_actif", data["modele_actif"])).strip()
            for nom in data["modeles"]:
                data["modeles"][nom]["accuracy"] = round(float(last.get(f"accuracy_{nom}", 0.5)), 4)
                data["modeles"][nom]["kappa"] = round(float(last.get(f"kappa_{nom}", 0.0)), 4)
                data["modeles"][nom]["latency_ms"] = round(float(last.get(f"latency_ms_{nom}", 0.0)), 3)
                # score_bandit absent des anciens CSV → fallback sur accuracy
                score_b = last.get(f"score_bandit_{nom}")
                if score_b is None or (isinstance(score_b, float) and pd.isna(score_b)):
                    score_b = last.get(f"accuracy_{nom}", 0.5)
                data["modeles"][nom]["score_bandit"] = round(float(score_b), 4)
            data["systeme"]["cpu_percent"] = float(last.get("cpu_percent", 0))
            data["systeme"]["ram_percent"] = float(last.get("ram_percent", 0))

    return data


@app.get("/history", summary="Historique complet des métriques")
async def history() -> Dict[str, Any]:
    """
    Retourne l'historique complet des métriques en JSON.

    Lit d'abord la mémoire interne du moteur ; si elle est vide (cas où
    l'API et le moteur tournent dans des processus séparés), lit le fichier
    results/history.csv écrit périodiquement par le moteur.

    Returns:
        Liste des entrées d'historique avec instance, précisions, etc.
    """
    _verifier_moteur()
    df = _MOTEUR.get_full_history()

    # Fallback : lire le CSV si la mémoire interne est vide
    # (moteur et API tournent dans des processus distincts)
    if df.empty:
        csv_path = Path(__file__).parent.parent / "results" / "history.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.DataFrame()

    if df.empty:
        return {"historique": [], "nb_entrees": 0}
    return {
        "historique": df.to_dict(orient="records"),
        "nb_entrees": len(df),
    }


@app.get("/models", summary="Liste des modèles candidats")
async def models() -> Dict[str, Any]:
    """
    Liste les modèles candidats et leurs scores bandit actuels.

    Si le moteur local est vide, lit les dernières valeurs depuis le CSV.

    Returns:
        Dictionnaire modèle → {score_bandit, accuracy, kappa, latency_ms}.
    """
    _verifier_moteur()

    modele_actif = _MOTEUR._modele_actif
    modeles_data = {
        nom: {
            "score_bandit": round(_MOTEUR.bandit.scores.get(nom, 0.5), 4),
            "count_bandit": _MOTEUR.bandit.counts.get(nom, 0),
            "accuracy": round(_MOTEUR.modeles[nom].get_accuracy(), 4),
            "kappa": round(_MOTEUR.modeles[nom].get_kappa(), 4),
            "latency_ms": round(_MOTEUR.modeles[nom].get_avg_latency(), 3),
        }
        for nom in _MOTEUR.modeles
    }

    # Fallback CSV si le moteur local est vide (processus séparés)
    if _MOTEUR._instance_count == 0:
        last = _lire_derniere_ligne_csv()
        if not last.empty:
            modele_actif = str(last.get("modele_actif", modele_actif)).strip()
            for nom in modeles_data:
                modeles_data[nom]["accuracy"] = round(float(last.get(f"accuracy_{nom}", 0.5)), 4)
                modeles_data[nom]["kappa"] = round(float(last.get(f"kappa_{nom}", 0.0)), 4)
                modeles_data[nom]["latency_ms"] = round(float(last.get(f"latency_ms_{nom}", 0.0)), 3)
                # score_bandit absent des anciens CSV → fallback sur accuracy
                score_b = last.get(f"score_bandit_{nom}")
                if score_b is None or (isinstance(score_b, float) and pd.isna(score_b)):
                    score_b = last.get(f"accuracy_{nom}", 0.5)
                modeles_data[nom]["score_bandit"] = round(float(score_b), 4)

    meilleur = max(modeles_data, key=lambda n: modeles_data[n]["score_bandit"])

    return {
        "modele_actif": modele_actif,
        "meilleur_modele": meilleur,
        "modeles": modeles_data,
    }


@app.post("/reset", summary="Réinitialiser le bandit (simulation de dérive)")
async def reset() -> Dict[str, str]:
    """
    Force un reset du bandit multi-bras.

    Simule une dérive manuelle : remet tous les scores à 0.5.

    Returns:
        Message de confirmation.
    """
    _verifier_moteur()
    _MOTEUR.bandit.reset_scores()
    logger.info("Reset manuel du bandit déclenché via API /reset")
    return {
        "status": "ok",
        "message": "Bandit réinitialisé (scores remis à 0.5)",
    }


@app.get("/changes", summary="Historique des changements de modèle")
async def changes() -> Dict[str, Any]:
    """
    Retourne l'historique des changements de modèle actif.

    Lit d'abord la mémoire interne ; fallback sur results/model_changes.csv
    si le moteur tourne dans un processus séparé.

    Returns:
        Liste des événements de changement (timestamp, ancien, nouveau, raison).
    """
    _verifier_moteur()
    df = _MOTEUR.get_model_changes()

    if df.empty:
        csv_path = Path(__file__).parent.parent / "results" / "model_changes.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.DataFrame()

    if df.empty:
        return {"changements": [], "nb_changements": 0}
    return {
        "changements": df.to_dict(orient="records"),
        "nb_changements": len(df),
    }
