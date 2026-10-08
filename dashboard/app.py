"""
app.py — Dashboard Streamlit temps réel pour le système Auto-ML Adaptatif.

Affiche en temps réel les métriques du moteur Auto-ML via l'API FastAPI.
Se rafraîchit toutes les 30 secondes.

Source unique de vérité : /history (CSV écrit périodiquement par le moteur).
Les cartes, le tableau et le graphique bandit extraient tous leurs données
depuis la dernière ligne de l'historique.

Lancement :
    streamlit run dashboard/app.py
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Configuration de la page ───────────────────────────────────────────────

st.set_page_config(
    page_title="Auto-ML Adaptatif — Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
MODELES = ["HT", "KNN", "SGD"]


# ─── Fonctions de chargement (TTL 30s) ──────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_history() -> Optional[pd.DataFrame]:
    """
    Récupère l'historique complet depuis /history.
    Source principale pour toutes les métriques numériques.
    """
    try:
        resp = httpx.get(f"{API_URL}/history", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if data["nb_entrees"] == 0:
            return None
        df = pd.DataFrame(data["historique"])
        # S'assurer que les colonnes numériques sont bien typées
        for col in df.columns:
            if col not in ("timestamp", "modele_actif"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_changes() -> Optional[pd.DataFrame]:
    """Récupère l'historique des changements de modèle depuis /changes."""
    try:
        resp = httpx.get(f"{API_URL}/changes", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if data["nb_changements"] == 0:
            return None
        return pd.DataFrame(data["changements"])
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_system() -> Dict[str, float]:
    """
    Récupère les métriques système temps réel (CPU, RAM) depuis /metrics.
    Seul endpoint utilisé pour les données système instantanées.
    """
    try:
        resp = httpx.get(f"{API_URL}/metrics", timeout=5.0)
        resp.raise_for_status()
        return resp.json().get("systeme", {})
    except Exception:
        return {}


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Auto-ML Adaptatif")
    st.markdown("**Système de recommandation dynamique de modèles**")
    st.markdown("---")
    st.markdown("🔄 Rafraîchissement : toutes les 30 secondes")
    st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

    # Vérification de la connexion à l'API
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=2.0)
        uptime = resp.json().get("uptime", "?")
        st.success(f"✅ API connectée (uptime: {uptime}s)")
    except Exception:
        st.error("❌ API non disponible — lancez l'API d'abord")
        st.code("uvicorn api.main:app --reload --port 8000")

    st.markdown("---")

    # Bouton de reset du bandit
    if st.button("🔄 Reset Bandit (simuler dérive)"):
        try:
            httpx.post(f"{API_URL}/reset", timeout=5.0)
            st.success("Bandit réinitialisé !")
            st.cache_data.clear()
        except Exception:
            st.error("Erreur lors du reset")

    st.markdown("---")
    st.markdown("**Modèles candidats :**")
    st.markdown("- 🌳 HT : Hoeffding Tree")
    st.markdown("- 🔍 KNN : K-Nearest Neighbors")
    st.markdown("- 📈 SGD : Régression Logistique")

# ─── Chargement des données ─────────────────────────────────────────────────

historique = fetch_history()
changements = fetch_changes()
systeme = fetch_system()

# Extraire la dernière ligne de l'historique (source principale des métriques)
derniere_ligne = historique.iloc[-1] if historique is not None else None

# ─── Section 1 : Métriques en direct ────────────────────────────────────────

st.title("🤖 Auto-ML Adaptatif — Tableau de Bord Temps Réel")
st.markdown("---")
st.subheader("📊 Métriques en direct")

if derniere_ligne is None:
    st.warning(
        "⚠️ Aucune donnée historique disponible. "
        "Lancez le moteur (`python -m src.automl_engine`) "
        "et l'API (`uvicorn api.main:app --reload --port 8000`)."
    )
else:
    modele_actif = str(derniere_ligne.get("modele_actif", "N/A")).strip()
    instance_count = int(derniere_ligne.get("instance", 0))
    nb_drifts = int(derniere_ligne.get("nb_drifts", 0))

    # Précision et latence du modèle actif
    accuracy_active = float(derniere_ligne.get(f"accuracy_{modele_actif}", 0.0))
    latency_active = float(derniere_ligne.get(f"latency_ms_{modele_actif}", 0.0))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🏆 Modèle actif", modele_actif)
    with col2:
        st.metric("🎯 Précision", f"{accuracy_active * 100:.2f}%")
    with col3:
        st.metric("⚡ Latence moy.", f"{latency_active:.3f} ms")
    with col4:
        st.metric("🌊 Dérives", nb_drifts)
    with col5:
        st.metric("🖥️ CPU", f"{systeme.get('cpu_percent', 0):.1f}%")
    with col6:
        st.metric("💾 RAM", f"{systeme.get('ram_percent', 0):.1f}%")

    st.markdown(f"*Instances traitées : {instance_count:,}*")

# ─── Section 2 : Précision glissante par modèle ─────────────────────────────

st.markdown("---")
st.subheader("📈 Précision glissante par modèle")

if historique is not None and not historique.empty:
    colonnes_acc = [f"accuracy_{m}" for m in MODELES if f"accuracy_{m}" in historique.columns]

    if colonnes_acc:
        df_plot = historique[["instance"] + colonnes_acc].copy()
        renommage = {c: c.replace("accuracy_", "") for c in colonnes_acc}
        df_plot = df_plot.rename(columns=renommage)

        df_melted = df_plot.melt(
            id_vars="instance",
            var_name="Modèle",
            value_name="Précision"
        )

        fig_acc = px.line(
            df_melted,
            x="instance",
            y="Précision",
            color="Modèle",
            title="Évolution de la précision glissante par modèle",
            labels={"instance": "Numéro d'instance", "Précision": "Précision"},
            color_discrete_map={"HT": "#2196F3", "KNN": "#FF9800", "SGD": "#4CAF50"},
        )

        # Lignes verticales rouges pour les dérives
        if changements is not None:
            drifts = changements[changements["raison"].str.startswith("derive", na=False)]
            for _, row in drifts.iterrows():
                fig_acc.add_vline(
                    x=row["instance"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Dérive ({row['raison']})",
                    annotation_position="top",
                )

        fig_acc.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
        )
        st.plotly_chart(fig_acc, use_container_width=True)
else:
    st.info("En attente de données historiques... Lancez le moteur Auto-ML.")

# ─── Section 3 : Scores bandit ──────────────────────────────────────────────

st.markdown("---")
st.subheader("🎰 Scores du Bandit Multi-Bras")

if derniere_ligne is not None:
    # Lire les scores bandit depuis la dernière ligne de l'historique.
    # Fallback sur accuracy si la colonne score_bandit est absente (ancien CSV).
    scores_bandit = {}
    for nom in MODELES:
        val = derniere_ligne.get(f"score_bandit_{nom}")
        if val is None or (isinstance(val, float) and val != val):  # None ou NaN
            val = derniere_ligne.get(f"accuracy_{nom}", 0.5)
        scores_bandit[nom] = float(val) if val is not None else 0.5

    modele_actif_str = str(derniere_ligne.get("modele_actif", ""))
    meilleur = max(scores_bandit, key=lambda n: scores_bandit[n])

    couleurs = [
        "#FF5722" if nom == meilleur else "#78909C"
        for nom in MODELES
    ]

    fig_bandit = go.Figure(go.Bar(
        x=[scores_bandit[n] for n in MODELES],
        y=MODELES,
        orientation="h",
        marker_color=couleurs,
        text=[f"{scores_bandit[n]:.4f}" for n in MODELES],
        textposition="outside",
    ))
    fig_bandit.update_layout(
        title=f"Score bandit par modèle (meilleur en rouge : {meilleur})",
        xaxis_title="Score bandit",
        yaxis_title="Modèle",
        height=300,
        xaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig_bandit, use_container_width=True)

    # Tableau détail des métriques — lu depuis dernière_ligne
    st.markdown("**Détail des métriques par modèle :**")
    rows = []
    for nom in MODELES:
        acc = float(derniere_ligne.get(f"accuracy_{nom}", 0.0))
        kappa = float(derniere_ligne.get(f"kappa_{nom}", 0.0))
        lat = float(derniere_ligne.get(f"latency_ms_{nom}", 0.0))
        score_b = scores_bandit[nom]
        est_actif = (nom == modele_actif_str)
        rows.append({
            "Modèle": f"{'⭐ ' if est_actif else ''}{nom}",
            "Précision": f"{acc * 100:.2f}%",
            "Kappa": f"{kappa:.4f}",
            "Latence (ms)": f"{lat:.3f}",
            "Score Bandit": f"{score_b:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── Section 4 : Historique des changements de modèle ──────────────────────

st.markdown("---")
st.subheader("🔄 Historique des changements de modèle")

if changements is not None and not changements.empty:
    df_affichage = changements.copy()
    df_affichage.columns = [c.replace("_", " ").title() for c in df_affichage.columns]
    st.dataframe(
        df_affichage.sort_values("Instance", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Aucun changement de modèle enregistré pour le moment.")

# ─── Section 5 : Monitoring système ─────────────────────────────────────────

st.markdown("---")
st.subheader("🖥️ Monitoring Système")

cpu = systeme.get("cpu_percent", 0)
ram = systeme.get("ram_percent", 0)
ram_mb = systeme.get("ram_utilisee_mb", 0)

col_cpu, col_ram = st.columns(2)

with col_cpu:
    fig_cpu = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cpu,
        title={"text": "CPU (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2196F3"},
            "steps": [
                {"range": [0, 50], "color": "#E3F2FD"},
                {"range": [50, 80], "color": "#FFF9C4"},
                {"range": [80, 100], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    fig_cpu.update_layout(height=300)
    st.plotly_chart(fig_cpu, use_container_width=True)

with col_ram:
    fig_ram = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ram,
        title={"text": f"RAM (%) — {ram_mb:.0f} MB utilisés"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF9800"},
            "steps": [
                {"range": [0, 60], "color": "#FFF3E0"},
                {"range": [60, 85], "color": "#FFE0B2"},
                {"range": [85, 100], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    fig_ram.update_layout(height=300)
    st.plotly_chart(fig_ram, use_container_width=True)

# ─── Courbe CPU/RAM historique depuis l'historique ──────────────────────────

if historique is not None and "cpu_percent" in historique.columns:
    df_sys = historique[["instance", "cpu_percent", "ram_percent"]].copy()
    df_sys_melted = df_sys.melt(
        id_vars="instance",
        var_name="Métrique",
        value_name="Valeur (%)"
    )
    df_sys_melted["Métrique"] = df_sys_melted["Métrique"].map({
        "cpu_percent": "CPU (%)",
        "ram_percent": "RAM (%)",
    })
    fig_sys = px.line(
        df_sys_melted,
        x="instance",
        y="Valeur (%)",
        color="Métrique",
        title="Évolution CPU et RAM au fil du temps",
        height=300,
        color_discrete_map={"CPU (%)": "#2196F3", "RAM (%)": "#FF9800"},
    )
    fig_sys.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_sys, use_container_width=True)

# ─── Rafraîchissement automatique ───────────────────────────────────────────

time.sleep(30)
st.rerun()
