"""
Agent d'analyse financiere -- version "rule-based" (sans LLM externe)
------------------------------------------------------------------
Cet agent comprend des questions en langage naturel sur les depenses
de l'utilisateur, et y repond a partir des donnees reellement
disponibles (jamais d'invention de chiffres).

IMPORTANT -- ce que cet agent EST et N'EST PAS :
- Ce n'est PAS un grand modele de langage (LLM) comme GPT ou Claude.
  Il fonctionne par reconnaissance de mots-cles et de motifs (regex),
  une approche "rule-based" (a base de regles).
- Il ne necessite AUCUNE cle API ni connexion internet : tout tourne
  en local, gratuitement.
- Il peut AIDER a structurer une saisie de depense (extraire montant,
  categorie, date d'une phrase), mais ne valide et n'enregistre jamais
  une donnee sans confirmation explicite de l'utilisateur -- l'agent
  analyse et propose, il n'agit jamais seul.

Cette architecture modulaire (une fonction d'entree, une fonction de
sortie) permettrait de remplacer le moteur "rule-based" par un vrai
appel a une API de LLM (Anthropic, OpenAI...) dans une version future,
sans changer le reste de l'application.
"""

import re
import pandas as pd

CATEGORIES_CONNUES = ["Logement", "Alimentation", "Transport", "Loisirs",
                       "Abonnements", "Santé", "Shopping", "Autre"]

MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}


def _trouver_categorie(texte):
    """Cherche une categorie connue mentionnee dans le texte (insensible a la casse)."""
    texte_lower = texte.lower()
    for cat in CATEGORIES_CONNUES:
        if cat.lower() in texte_lower:
            return cat
    return None


def _trouver_mois(texte):
    """Cherche un nom de mois mentionne dans le texte, renvoie son numero (1-12)."""
    texte_lower = texte.lower()
    for nom, numero in MOIS_FR.items():
        if nom in texte_lower:
            return numero
    return None


def _trouver_montant(texte):
    """Extrait un montant numerique (ex: '45.50', '45,50', '45') du texte."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:€|eur|euros?)?", texte)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


# ----------------------------------------------------------------------
# INTENTIONS RECONNUES : chaque fonction repond a un type de question
# ----------------------------------------------------------------------
def _repondre_total_categorie(df_depenses, categorie, mois=None, annee=None):
    sous_df = df_depenses[df_depenses["categorie"] == categorie]
    if mois:
        sous_df = sous_df[sous_df["date"].dt.month == mois]
    if annee:
        sous_df = sous_df[sous_df["date"].dt.year == annee]

    total = sous_df["montant"].sum()
    nb = len(sous_df)
    if nb == 0:
        return f"Aucune dépense enregistrée pour « {categorie} » sur cette période."
    periode = f" en {[k for k,v in MOIS_FR.items() if v == mois][0].capitalize()}" if mois else ""
    return f"Tu as dépensé **{total:.2f} €** en « {categorie} »{periode} ({nb} dépense{'s' if nb > 1 else ''})."


def _repondre_total_mois_actuel(df_depenses):
    mois_actuel = pd.Timestamp.now().to_period("M")
    sous_df = df_depenses[df_depenses["date"].dt.to_period("M") == mois_actuel]
    total = sous_df["montant"].sum()
    if total == 0:
        return "Aucune dépense enregistrée ce mois-ci pour le moment."
    return f"Tu as dépensé **{total:.2f} €** au total ce mois-ci, sur {len(sous_df)} dépenses."


def _repondre_categorie_plus_chere(df_depenses):
    if df_depenses.empty:
        return "Pas encore de données pour identifier ta catégorie la plus coûteuse."
    totaux = df_depenses.groupby("categorie")["montant"].sum().sort_values(ascending=False)
    cat_top = totaux.index[0]
    montant_top = totaux.iloc[0]
    return f"Ta catégorie la plus coûteuse est **{cat_top}**, avec {montant_top:.2f} € dépensés au total."


def _repondre_moyenne_categorie(df_depenses, categorie):
    sous_df = df_depenses[df_depenses["categorie"] == categorie].copy()
    if sous_df.empty:
        return f"Aucune dépense enregistrée pour « {categorie} »."
    sous_df["mois"] = sous_df["date"].dt.to_period("M")
    moyenne = sous_df.groupby("mois")["montant"].sum().mean()
    return f"En moyenne, tu dépenses **{moyenne:.2f} €/mois** en « {categorie} »."


def _proposer_saisie_depense(texte):
    """
    Analyse une phrase qui ressemble a une description de depense
    (ex: "j'ai depense 45 euros en courses hier") et propose une
    structuration -- SANS jamais l'enregistrer automatiquement.
    """
    montant = _trouver_montant(texte)
    categorie = _trouver_categorie(texte)

    if montant is None:
        return ("Je n'ai pas réussi à identifier de montant dans ta phrase. "
                "Essaie par exemple : « j'ai dépensé 45€ en Alimentation ».")

    categorie_affichee = categorie or "non identifiée — précise une catégorie parmi : " + ", ".join(CATEGORIES_CONNUES)
    return (f"J'ai compris : **{montant:.2f} €**, catégorie **{categorie_affichee}**.\n\n"
            f"⚠️ Je ne l'enregistre pas automatiquement — vérifie ces informations et "
            f"utilise le formulaire « Ajouter une dépense » pour confirmer toi-même la saisie.")


# ----------------------------------------------------------------------
# ROUTEUR PRINCIPAL : determine l'intention de la question posee
# ----------------------------------------------------------------------
def repondre(question, df_depenses):
    """
    Point d'entree principal de l'agent. Prend une question en texte
    libre et le DataFrame des depenses, renvoie une reponse textuelle.
    Ne modifie jamais les donnees.
    """
    if df_depenses is None or df_depenses.empty:
        return "Aucune dépense n'est encore enregistrée — je n'ai rien à analyser pour l'instant."

    q = question.lower().strip()
    est_une_question = "?" in q or q.startswith(("combien", "quelle", "quel", "quand", "moyenne"))

    # Intention : aide a structurer une saisie ("j'ai dépensé...", "j'ai payé...")
    # Uniquement si ce n'est PAS une question (sinon "combien j'ai dépensé en X ?"
    # serait a tort interprete comme une nouvelle depense a structurer)
    if not est_une_question and any(motif in q for motif in ["j'ai dépensé", "j'ai depense", "j'ai payé", "j'ai paye", "j'ai acheté", "j'ai achete"]):
        return _proposer_saisie_depense(question)

    # Intention : quelle est ma catégorie la plus chère
    if any(motif in q for motif in ["plus cher", "plus chère", "plus coûteuse", "plus couteuse", "le plus dépensé", "le plus depense"]):
        return _repondre_categorie_plus_chere(df_depenses)

    # Intention : moyenne mensuelle d'une catégorie
    if "moyenne" in q:
        categorie = _trouver_categorie(q)
        if categorie:
            return _repondre_moyenne_categorie(df_depenses, categorie)
        return "Précise une catégorie pour que je calcule sa moyenne mensuelle (ex : « moyenne Alimentation »)."

    # Intention : total sur une catégorie (avec ou sans mois précisé)
    categorie = _trouver_categorie(q)
    if categorie:
        mois = _trouver_mois(q)
        return _repondre_total_categorie(df_depenses, categorie, mois=mois)

    # Intention : total du mois en cours (question générale)
    if any(motif in q for motif in ["combien", "total", "dépensé", "depense"]):
        return _repondre_total_mois_actuel(df_depenses)

    return ("Je n'ai pas compris cette question. Tu peux me demander par exemple :\n\n"
            "- « Combien j'ai dépensé en Loisirs ce mois-ci ? »\n"
            "- « Quelle est ma catégorie la plus chère ? »\n"
            "- « Moyenne Alimentation »\n"
            "- « J'ai dépensé 20€ en Transport » (je t'aide à structurer, sans rien enregistrer)")
