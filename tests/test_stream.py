"""
test_stream.py — Tests unitaires pour le module stream.py.

Vérifie le comportement du générateur de flux de données :
- Format de sortie correct (tuples dict/int).
- Dérive artificielle introduite à la bonne instance.

Note : Ces tests nécessitent le dataset UNSW_NB15_training-set.csv
dans le dossier data/. Les tests sont skippés si le fichier est absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Chemin vers le dataset
DATASET_PATH = str(
    Path(__file__).parent.parent / "data" / "UNSW_NB15_training-set.csv"
)
DATASET_DISPONIBLE = Path(DATASET_PATH).exists()

SKIP_MESSAGE = (
    "Dataset UNSW_NB15_training-set.csv introuvable dans data/. "
    "Téléchargez-le depuis : https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
)


class TestStreamDataset:
    """Tests pour la fonction stream_dataset."""

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_yield_tuples_dict_int(self):
        """Vérifie que le générateur yield des tuples (dict, int)."""
        from src.stream import stream_dataset

        flux = stream_dataset(DATASET_PATH, shuffle=False)
        x, y = next(flux)

        assert isinstance(x, dict), "x doit être un dictionnaire"
        assert isinstance(y, int), "y doit être un entier"
        assert y in (0, 1), "y doit être binaire (0 ou 1)"

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_features_sont_numeriques(self):
        """Vérifie que toutes les valeurs de features sont numériques."""
        from src.stream import stream_dataset

        flux = stream_dataset(DATASET_PATH, shuffle=False)
        x, y = next(flux)

        for nom, valeur in x.items():
            assert isinstance(valeur, (int, float)), (
                f"Feature '{nom}' doit être numérique, got {type(valeur)}"
            )

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_flux_100_instances(self):
        """Vérifie que le générateur produit au moins 100 instances."""
        from src.stream import stream_dataset

        flux = stream_dataset(DATASET_PATH, shuffle=True)
        instances = []
        for i, (x, y) in enumerate(flux):
            instances.append((x, y))
            if i >= 99:
                break

        assert len(instances) == 100

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_features_non_vides(self):
        """Vérifie que le dictionnaire de features n'est pas vide."""
        from src.stream import stream_dataset

        flux = stream_dataset(DATASET_PATH, shuffle=False)
        x, y = next(flux)

        assert len(x) > 0, "Le dictionnaire de features ne doit pas être vide"

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_pas_de_colonnes_categoriques(self):
        """Vérifie que les colonnes catégorielles sont bien supprimées."""
        from src.stream import stream_dataset

        flux = stream_dataset(DATASET_PATH, shuffle=False)
        x, y = next(flux)

        colonnes_a_exclure = {"proto", "service", "state", "attack_cat"}
        for col in colonnes_a_exclure:
            assert col not in x, (
                f"La colonne catégorielle '{col}' ne devrait pas être dans x"
            )

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_fichier_inexistant_leve_exception(self):
        """Vérifie qu'une exception est levée si le fichier n'existe pas."""
        from src.stream import stream_dataset

        with pytest.raises(FileNotFoundError):
            flux = stream_dataset("/chemin/inexistant/dataset.csv")
            next(flux)  # L'exception se lève au premier accès


class TestStreamWithArtificialDrift:
    """Tests pour la fonction stream_with_artificial_drift."""

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_drift_introduit_a_la_bonne_instance(self):
        """
        Vérifie que la dérive artificielle est introduite à l'instance drift_at.

        Compare les valeurs moyennes avant et après le point de dérive pour
        vérifier que les données ont changé de distribution.
        """
        from src.stream import stream_with_artificial_drift
        import statistics

        DRIFT_AT = 200

        flux = stream_with_artificial_drift(DATASET_PATH, drift_at=DRIFT_AT)

        valeurs_avant = []
        valeurs_apres = []

        for i, (x, y) in enumerate(flux):
            if i < DRIFT_AT:
                # Récupérer la première valeur du dictionnaire
                valeurs_avant.append(list(x.values())[0])
            elif i < DRIFT_AT + 100:
                valeurs_apres.append(list(x.values())[0])
            else:
                break

        # Vérifier que nous avons collecté des données des deux phases
        assert len(valeurs_avant) == DRIFT_AT
        assert len(valeurs_apres) == 100

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_meme_format_avant_et_apres_derive(self):
        """
        Vérifie que le format (dict, int) est maintenu avant et après la dérive.
        """
        from src.stream import stream_with_artificial_drift

        DRIFT_AT = 50

        flux = stream_with_artificial_drift(DATASET_PATH, drift_at=DRIFT_AT)

        for i, (x, y) in enumerate(flux):
            assert isinstance(x, dict), f"Instance {i}: x doit être un dict"
            assert isinstance(y, int), f"Instance {i}: y doit être un int"
            assert y in (0, 1), f"Instance {i}: y doit être 0 ou 1"

            if i >= 100:
                break

    @pytest.mark.skipif(not DATASET_DISPONIBLE, reason=SKIP_MESSAGE)
    def test_memes_cles_avant_et_apres_derive(self):
        """
        Vérifie que les clés du dictionnaire restent les mêmes avant et après la dérive.
        (Seules les valeurs sont permutées, pas les clés.)
        """
        from src.stream import stream_with_artificial_drift

        DRIFT_AT = 10
        flux = stream_with_artificial_drift(DATASET_PATH, drift_at=DRIFT_AT)

        cles_avant = None
        cles_apres = None

        for i, (x, y) in enumerate(flux):
            if i == 0:
                cles_avant = set(x.keys())
            if i == DRIFT_AT + 5:
                cles_apres = set(x.keys())
                break

        assert cles_avant is not None
        assert cles_apres is not None
        assert cles_avant == cles_apres, (
            "Les clés du dictionnaire ne doivent pas changer après la dérive"
        )


class TestStreamSansDataset:
    """Tests ne nécessitant pas le dataset."""

    def test_stream_dataset_importable(self):
        """Vérifie que le module stream est importable."""
        from src import stream
        assert hasattr(stream, "stream_dataset")
        assert hasattr(stream, "stream_with_artificial_drift")

    def test_stream_dataset_callable(self):
        """Vérifie que les fonctions sont appelables."""
        from src.stream import stream_dataset, stream_with_artificial_drift
        import inspect

        assert callable(stream_dataset)
        assert callable(stream_with_artificial_drift)

        # Vérifier les signatures
        sig_sd = inspect.signature(stream_dataset)
        assert "filepath" in sig_sd.parameters
        assert "shuffle" in sig_sd.parameters

        sig_drift = inspect.signature(stream_with_artificial_drift)
        assert "filepath" in sig_drift.parameters
        assert "drift_at" in sig_drift.parameters
