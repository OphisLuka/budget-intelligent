"""
Budget personnel intelligent
------------------------------------------------------------------
Application de suivi de budget : saisie de dépenses, alertes de
dépassement, prédiction ML par catégorie (régression linéaire
scikit-learn), et un agent d'analyse en langage naturel (rule-based,
sans API externe -- voir agent.py).

Lancement : streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

from agent import repondre as agent_repondre

FICHIER_DEPENSES = "depenses.csv"
FICHIER_BUDGETS = "budgets.csv"
COLONNES_DEPENSES = ["date", "categorie", "montant"]
CATEGORIES = ["Logement", "Alimentation", "Transport", "Loisirs",
              "Abonnements", "Santé", "Shopping", "Autre"]
MOIS_MIN_POUR_REGRESSION = 4

# --- Palette du thème visuel ---
VIOLET = "#A855F7"
CYAN = "#00F0FF"
VERT = "#39FF6A"
ROUGE = "#FF3860"
FOND = "#0A0A12"
FOND_CARTE = "#13131F"

st.set_page_config(page_title="BUDGET//SYS", layout="wide", page_icon="🟣")


# ----------------------------------------------------------------------
# THEME VISUEL (CSS injecte)
# ----------------------------------------------------------------------
def injecter_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap');

    .stApp {{
        background: {FOND};
        background-image:
            radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(0, 240, 255, 0.06) 0%, transparent 40%);
    }}

    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        color: #E0E0F0;
    }}

    h1, h2, h3 {{
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.02em;
    }}

    h1 {{
        color: {VIOLET} !important;
        text-shadow: 0 0 18px rgba(168, 85, 247, 0.45);
        border-bottom: 1px solid rgba(168, 85, 247, 0.3);
        padding-bottom: 0.4em;
    }}

    h1::after {{
        content: " _";
        color: {CYAN};
        animation: clignote 1.1s steps(1) infinite;
    }}

    @keyframes clignote {{
        50% {{ opacity: 0; }}
    }}

    h2, h3 {{
        color: {CYAN} !important;
    }}

    /* Cartes / conteneurs */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {FOND_CARTE};
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 4px;
    }}

    /* Barre laterale */
    [data-testid="stSidebar"] {{
        background: #0D0D18;
        border-right: 1px solid rgba(0, 240, 255, 0.2);
    }}

    /* Boutons */
    .stButton button, .stFormSubmitButton button {{
        background: transparent;
        border: 1px solid {VIOLET};
        color: {VIOLET};
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-radius: 2px;
        transition: all 0.15s ease;
    }}

    .stButton button:hover, .stFormSubmitButton button:hover {{
        background: {VIOLET};
        color: {FOND};
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.6);
    }}

    /* Metriques */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        color: {CYAN} !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        font-size: 0.75em;
        letter-spacing: 0.08em;
        color: #8888AA !important;
    }}

    /* Inputs */
    input, textarea, select, .stSelectbox div[data-baseweb="select"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Tableaux */
    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(168, 85, 247, 0.2);
    }}

    /* Bandeau de scanline discret en haut de page */
    .scanline-header {{
        font-family: 'JetBrains Mono', monospace;
        color: {CYAN};
        font-size: 0.75em;
        letter-spacing: 0.15em;
        opacity: 0.55;
        border-bottom: 1px dashed rgba(0, 240, 255, 0.25);
        padding-bottom: 0.5em;
        margin-bottom: 0.5em;
    }}
    </style>
    """, unsafe_allow_html=True)


injecter_theme()
st.markdown('<p class="scanline-header">SYS::BUDGET // SESSION ACTIVE // MODE LOCAL — AUCUNE DONNÉE TRANSMISE</p>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# FONCTIONS UTILITAIRES : depenses
# ----------------------------------------------------------------------
def charger_depenses():
    if os.path.exists(FICHIER_DEPENSES):
        df = pd.read_csv(FICHIER_DEPENSES, dtype=str)
        df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date", "montant"])
    df_vide = pd.DataFrame(columns=COLONNES_DEPENSES)
    df_vide["date"] = pd.to_datetime(df_vide["date"])
    return df_vide


def sauvegarder_depenses(df):
    df_a_sauver = df.copy()
    df_a_sauver["date"] = df_a_sauver["date"].dt.strftime("%Y-%m-%d")
    df_a_sauver.to_csv(FICHIER_DEPENSES, index=False)


def ajouter_depense(date, categorie, montant):
    df = charger_depenses()
    nouvelle = pd.DataFrame([{"date": pd.Timestamp(date), "categorie": categorie, "montant": montant}])
    df = pd.concat([df, nouvelle], ignore_index=True)
    sauvegarder_depenses(df)


# ----------------------------------------------------------------------
# FONCTIONS UTILITAIRES : budgets
# ----------------------------------------------------------------------
def charger_budgets():
    if os.path.exists(FICHIER_BUDGETS):
        df = pd.read_csv(FICHIER_BUDGETS, dtype=str)
        df["budget_mensuel"] = pd.to_numeric(df["budget_mensuel"], errors="coerce")
        return df
    return pd.DataFrame(columns=["categorie", "budget_mensuel"])


def sauvegarder_budget(categorie, budget_mensuel):
    df = charger_budgets()
    df = df[df["categorie"] != categorie]
    if budget_mensuel is not None:
        nouvelle = pd.DataFrame([{"categorie": categorie, "budget_mensuel": budget_mensuel}])
        df = pd.concat([df, nouvelle], ignore_index=True)
    df.to_csv(FICHIER_BUDGETS, index=False)


def obtenir_budget_effectif(categorie, df_budgets, df_depenses):
    ligne = df_budgets[df_budgets["categorie"] == categorie]
    if not ligne.empty and pd.notna(ligne.iloc[0]["budget_mensuel"]):
        return float(ligne.iloc[0]["budget_mensuel"]), False

    historique_cat = df_depenses[df_depenses["categorie"] == categorie].copy()
    if historique_cat.empty:
        return None, True
    historique_cat["mois"] = historique_cat["date"].dt.to_period("M")
    moyenne = historique_cat.groupby("mois")["montant"].sum().mean()
    return float(moyenne), True


# ----------------------------------------------------------------------
# PREDICTION ML
# ----------------------------------------------------------------------
def predire_depenses_categorie(df_depenses, categorie):
    historique_cat = df_depenses[df_depenses["categorie"] == categorie].copy()
    if historique_cat.empty:
        return None, "aucune donnée"

    historique_cat["mois"] = historique_cat["date"].dt.to_period("M")
    totaux_par_mois = historique_cat.groupby("mois")["montant"].sum().sort_index()

    if len(totaux_par_mois) < MOIS_MIN_POUR_REGRESSION:
        return float(totaux_par_mois.mean()), "moyenne (historique encore court)"

    X = np.arange(len(totaux_par_mois)).reshape(-1, 1)
    y = totaux_par_mois.values
    modele = LinearRegression()
    modele.fit(X, y)
    mois_suivant = np.array([[len(totaux_par_mois)]])
    prediction = max(0, modele.predict(mois_suivant)[0])
    return float(prediction), "régression linéaire"


# ----------------------------------------------------------------------
# BARRE LATERALE
# ----------------------------------------------------------------------
st.sidebar.markdown("### ▸ NOUVELLE ENTRÉE")
with st.sidebar.form("formulaire_depense", clear_on_submit=True):
    date_depense = st.date_input("Date")
    categorie = st.selectbox("Catégorie", CATEGORIES)
    montant = st.number_input("Montant (€)", min_value=0.0, step=1.0, value=0.0)
    valider = st.form_submit_button("⏎ ENREGISTRER")

    if valider:
        if montant <= 0:
            st.sidebar.error("ERR :: montant invalide")
        else:
            ajouter_depense(date_depense, categorie, montant)
            st.sidebar.success("OK :: dépense enregistrée")

st.sidebar.markdown("---")
st.sidebar.markdown("### ▸ DÉFINIR UN BUDGET")
with st.sidebar.form("formulaire_budget", clear_on_submit=True):
    categorie_budget = st.selectbox("Catégorie", CATEGORIES, key="cat_budget")
    montant_budget = st.number_input(
        "Budget mensuel (€) — 0 = auto",
        min_value=0.0, step=10.0, value=0.0
    )
    valider_budget = st.form_submit_button("⏎ DÉFINIR")

    if valider_budget:
        sauvegarder_budget(categorie_budget, montant_budget if montant_budget > 0 else None)
        st.sidebar.success(f"OK :: budget mis à jour")

st.sidebar.markdown("---")
st.sidebar.caption("⚙ Données stockées localement (depenses.csv, budgets.csv). Aucune transmission externe.")

# ----------------------------------------------------------------------
# CONTENU PRINCIPAL
# ----------------------------------------------------------------------
st.title("BUDGET//SYS")
st.caption("Suivi de dépenses, alertes de seuil, prédiction par régression — exécution 100% locale.")

df_depenses = charger_depenses()
df_budgets = charger_budgets()

if df_depenses.empty:
    st.info("◇ Aucune donnée en mémoire. Initialise une première entrée via le panneau latéral.")
else:
    mois_actuel = pd.Timestamp.now().to_period("M")
    depenses_mois_actuel = df_depenses[df_depenses["date"].dt.to_period("M") == mois_actuel]

    total_mois = depenses_mois_actuel["montant"].sum()
    nb_depenses = len(df_depenses)
    nb_mois_historique = df_depenses["date"].dt.to_period("M").nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("DÉPENSES // MOIS COURANT", f"{total_mois:.2f} €")
    col2.metric("ENTRÉES TOTALES", nb_depenses)
    col3.metric("MOIS D'HISTORIQUE", nb_mois_historique)

    st.markdown("---")

    # --- ALERTES ---
    st.subheader("⚠ ALERTES // SEUIL DE BUDGET")
    une_alerte_affichee = False
    for cat in CATEGORIES:
        depense_cat = depenses_mois_actuel[depenses_mois_actuel["categorie"] == cat]["montant"].sum()
        if depense_cat == 0:
            continue
        budget_effectif, est_auto = obtenir_budget_effectif(cat, df_budgets, df_depenses)
        if budget_effectif is None:
            continue

        ratio = depense_cat / budget_effectif if budget_effectif > 0 else 0
        source_budget = "moyenne auto" if est_auto else "budget fixé"

        if ratio >= 1.0:
            st.error(f"🔴 **{cat}** :: {depense_cat:.2f}€ / {budget_effectif:.2f}€ ({source_budget}) "
                      f"— DÉPASSEMENT +{(ratio - 1) * 100:.0f}%")
            une_alerte_affichee = True
        elif ratio >= 0.8:
            st.warning(f"🟡 **{cat}** :: {depense_cat:.2f}€ / {budget_effectif:.2f}€ ({source_budget}) "
                        f"— {ratio * 100:.0f}% du seuil")
            une_alerte_affichee = True

    if not une_alerte_affichee:
        st.success("🟢 STATUS :: aucun seuil franchi ce mois-ci.")

    st.markdown("---")

    # --- GRAPHIQUES (theme sombre pour matplotlib) ---
    plt.rcParams.update({
        "figure.facecolor": FOND_CARTE,
        "axes.facecolor": FOND_CARTE,
        "axes.edgecolor": "#8888AA",
        "axes.labelcolor": "#E0E0F0",
        "text.color": "#E0E0F0",
        "xtick.color": "#8888AA",
        "ytick.color": "#8888AA",
        "grid.color": "#2A2A3A",
    })

    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("▸ ÉVOLUTION MENSUELLE")
        df_depenses["mois"] = df_depenses["date"].dt.to_period("M").astype(str)
        totaux_mensuels = df_depenses.groupby("mois")["montant"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        totaux_mensuels.plot(kind="bar", ax=ax, color=VIOLET)
        ax.set_ylabel("€")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with col_droite:
        st.subheader("▸ RÉPARTITION PAR CATÉGORIE")
        totaux_categorie = df_depenses.groupby("categorie")["montant"].sum().sort_values(ascending=False)
        couleurs_camembert = [VIOLET, CYAN, VERT, "#FF8C42", "#FF3860", "#8888FF", "#FFD700", "#888888"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(totaux_categorie.values, labels=totaux_categorie.index, autopct="%1.0f%%",
               colors=couleurs_camembert, textprops={"color": "#E0E0F0"})
        st.pyplot(fig)

    st.markdown("---")

    # --- PREDICTION ML ---
    st.subheader("▸ PROJECTION // MOIS SUIVANT")
    st.caption(
        f"Régression linéaire (scikit-learn) si historique ≥ {MOIS_MIN_POUR_REGRESSION} mois, "
        "sinon repli sur moyenne simple."
    )

    lignes_prediction = []
    for cat in CATEGORIES:
        if df_depenses[df_depenses["categorie"] == cat].empty:
            continue
        prediction, methode = predire_depenses_categorie(df_depenses, cat)
        budget_effectif, _ = obtenir_budget_effectif(cat, df_budgets, df_depenses)
        lignes_prediction.append({
            "Catégorie": cat,
            "Projection": f"{prediction:.2f} €",
            "Méthode": methode,
            "Référence": f"{budget_effectif:.2f} €" if budget_effectif else "—",
        })

    if lignes_prediction:
        st.dataframe(pd.DataFrame(lignes_prediction), use_container_width=True, hide_index=True)
    else:
        st.caption("Données insuffisantes pour générer une projection.")

    st.markdown("---")

    # --- AGENT IA ---
    st.subheader("🤖 AGENT // INTERROGATION EN LANGAGE NATUREL")
    st.caption(
        "Agent local à base de règles (sans LLM externe, aucune clé API requise). "
        "Il analyse et répond à partir de tes données réelles — il ne crée et "
        "n'enregistre jamais rien lui-même."
    )

    question = st.text_input(
        "Pose ta question",
        placeholder="Ex: Combien j'ai dépensé en Loisirs ce mois-ci ?",
        key="question_agent"
    )
    if question:
        reponse = agent_repondre(question, df_depenses)
        st.markdown(f"> {reponse}")

    with st.expander("Exemples de questions possibles"):
        st.markdown("""
        - « Combien j'ai dépensé en Alimentation ? »
        - « Combien en Transport en mars ? »
        - « Quelle est ma catégorie la plus chère ? »
        - « Moyenne Loisirs »
        - « J'ai dépensé 20€ en Shopping » *(aide à la saisie, sans enregistrement automatique)*
        """)

    st.markdown("---")
    st.subheader("▸ LOG // HISTORIQUE COMPLET")
    st.dataframe(
        df_depenses[["date", "categorie", "montant"]].sort_values("date", ascending=False),
        use_container_width=True, hide_index=True
    )

# ----------------------------------------------------------------------
# IMPORT CSV
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("▸ IMPORT // FICHIER CSV")
st.caption("Colonnes attendues : date, categorie, montant")
fichier_importe = st.file_uploader("Sélectionner un fichier", type="csv")
if fichier_importe is not None:
    df_importe = pd.read_csv(fichier_importe, dtype=str)
    df_importe["montant"] = pd.to_numeric(df_importe["montant"], errors="coerce")
    df_importe["date"] = pd.to_datetime(df_importe["date"], errors="coerce")
    df_importe = df_importe.dropna(subset=["date", "montant"])

    df_existant = charger_depenses()
    df_final = pd.concat([df_existant, df_importe], ignore_index=True).drop_duplicates()
    sauvegarder_depenses(df_final)
    st.success(f"OK :: {len(df_importe)} entrées importées.")
    st.rerun()
