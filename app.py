"""
Budget personnel intelligent — v5
- Dépenses récurrentes (toutes catégories) vs ponctuelles
- Fréquence : hebdo / mensuel / annuel avec calcul pro-rata exact
- Catégories personnalisables
- Modale image au clic
- Logo SVG sidebar
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json, base64, calendar
from datetime import date, timedelta
from pathlib import Path
from sklearn.linear_model import LinearRegression
from agent import repondre as agent_repondre

# ── CONSTANTES ────────────────────────────────────────────────────────
FICHIER_DEPENSES   = "depenses.csv"
FICHIER_BUDGETS    = "budgets.csv"
FICHIER_RECURRENTS = "recurrents.json"
FICHIER_CATEGORIES = "categories.json"
DOSSIER_PHOTOS     = Path("photos_depenses")
DOSSIER_PHOTOS.mkdir(exist_ok=True)

CATEGORIES_DEFAULT = ["Logement","Alimentation","Transport",
                      "Loisirs","Santé","Shopping","Autre"]
ICONES_DEFAULT = {
    "Logement":"🏠","Alimentation":"🍽️","Transport":"🚇",
    "Loisirs":"🎮","Santé":"💊","Shopping":"🛍️","Autre":"📦",
}
ICONES_CUSTOM_DEFAULT = "🔷"
FREQUENCES = ["Mensuel","Hebdomadaire","Annuel"]

VIOLET="#A855F7"; CYAN="#00E5FF"; VERT="#39FF6A"
ROUGE="#FF3860"; OR="#FFD700";   FOND="#0A0A12"

# ── PAGE CONFIG ───────────────────────────────────────────────────────
from PIL import Image as PILImage
_favicon = PILImage.open("favicon.png") if os.path.exists("favicon.png") else "🟣"
st.set_page_config(page_title="BUDGET//SYS", layout="wide", page_icon=_favicon)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap');
.stApp{{background:{FOND};background-image:
    radial-gradient(circle at 12% 8%,rgba(168,85,247,.09) 0%,transparent 38%),
    radial-gradient(circle at 88% 92%,rgba(0,229,255,.06) 0%,transparent 38%);}}
html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;color:#E0E0F0;}}
h1,h2,h3{{font-family:'JetBrains Mono',monospace!important;letter-spacing:.02em;}}
h1{{color:{VIOLET}!important;text-shadow:0 0 18px rgba(168,85,247,.4);
    border-bottom:1px solid rgba(168,85,247,.25);padding-bottom:.4em;}}
h2,h3{{color:{CYAN}!important;}}
.inv-card{{background:linear-gradient(145deg,rgba(16,16,28,.95),rgba(24,14,42,.9));
    border:1px solid rgba(168,85,247,.3);border-radius:6px;
    padding:14px 12px 11px;text-align:center;position:relative;
    overflow:hidden;margin-bottom:4px;}}
.inv-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,229,255,.55),transparent);}}
.inv-card.ok     {{border-color:rgba(168,85,247,.3);}}
.inv-card.warn   {{border-color:{OR};box-shadow:0 0 12px rgba(255,215,0,.2);}}
.inv-card.alert  {{border-color:{ROUGE};box-shadow:0 0 12px rgba(255,56,96,.25);}}
.inv-icone{{font-size:1.9em;line-height:1.2;}}
.inv-nom{{font-family:'JetBrains Mono',monospace;font-size:.68em;
    letter-spacing:.09em;color:#7777AA;text-transform:uppercase;margin:5px 0 7px;}}
.inv-xz{{font-family:'JetBrains Mono',monospace;font-size:1em;
    font-weight:600;color:{CYAN};}}
.inv-bar-bg{{background:rgba(255,255,255,.07);border-radius:3px;
    height:4px;margin-top:9px;overflow:hidden;}}
.inv-bar{{height:100%;border-radius:3px;}}
.rec-badge{{font-family:'JetBrains Mono',monospace;font-size:.55em;
    color:{CYAN};border:1px solid rgba(0,229,255,.4);
    border-radius:2px;padding:1px 5px;margin-left:4px;}}
.rec-off{{color:#FF8C42;border-color:rgba(255,140,66,.4);}}
[data-testid="stSidebar"]{{background:#0B0B17;
    border-right:1px solid rgba(0,229,255,.15);}}
.sidebar-sep{{height:1px;background:linear-gradient(90deg,
    transparent,rgba(0,229,255,.2),transparent);margin:10px 0;}}
.logo-sub{{font-family:'JetBrains Mono',monospace;font-size:.6em;
    color:rgba(0,229,255,.45);letter-spacing:.2em;text-align:center;
    margin-bottom:4px;}}
.stButton button,.stFormSubmitButton button{{
    background:transparent;border:1px solid {VIOLET};color:{VIOLET};
    font-family:'JetBrains Mono',monospace;text-transform:uppercase;
    letter-spacing:.05em;border-radius:2px;transition:all .15s;}}
.stButton button:hover,.stFormSubmitButton button:hover{{
    background:{VIOLET};color:{FOND};box-shadow:0 0 14px rgba(168,85,247,.55);}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace!important;
    color:{CYAN}!important;text-shadow:0 0 8px rgba(0,229,255,.35);}}
[data-testid="stMetricLabel"]{{font-family:'JetBrains Mono',monospace!important;
    font-size:.72em!important;letter-spacing:.08em;
    text-transform:uppercase;color:#6666AA!important;}}
input,textarea,select{{font-family:'JetBrains Mono',monospace!important;}}
[data-testid="stDataFrame"]{{border:1px solid rgba(168,85,247,.18);}}
.scan{{font-family:'JetBrains Mono',monospace;font-size:.7em;
    letter-spacing:.14em;color:rgba(0,229,255,.4);
    border-bottom:1px dashed rgba(0,229,255,.18);padding-bottom:.5em;}}
.photo-thumb{{border-radius:3px;border:1px solid rgba(0,229,255,.25);
    max-width:72px;max-height:54px;object-fit:cover;cursor:pointer;}}
</style>
""", unsafe_allow_html=True)

# ── FONCTIONS : CATÉGORIES ────────────────────────────────────────────
def charger_categories():
    if os.path.exists(FICHIER_CATEGORIES):
        with open(FICHIER_CATEGORIES,"r",encoding="utf-8") as f:
            return json.load(f)
    return {"categories": CATEGORIES_DEFAULT, "icones": ICONES_DEFAULT}

def sauvegarder_categories(data):
    with open(FICHIER_CATEGORIES,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def get_categories():
    d = charger_categories()
    return d["categories"], d.get("icones", ICONES_DEFAULT)

# ── FONCTIONS : RÉCURRENTS ────────────────────────────────────────────
def charger_recurrents():
    """
    Format : {id: {nom, categorie, montant, frequence,
                   date_debut, actif, dernier_mois}}
    dernier_mois=True → encore prélevé ce mois, arrêté après
    """
    if not os.path.exists(FICHIER_RECURRENTS): return {}
    with open(FICHIER_RECURRENTS,"r",encoding="utf-8") as f:
        return json.load(f)

def sauvegarder_recurrents(d):
    with open(FICHIER_RECURRENTS,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False,indent=2)

def montant_mensuel(montant, frequence, annee, mois_num):
    """Calcule le montant équivalent mensuel selon la fréquence et les jours du mois."""
    nb_jours = calendar.monthrange(annee, mois_num)[1]
    if frequence == "Mensuel":
        return float(montant)
    elif frequence == "Hebdomadaire":
        # Nombre exact de semaines dans ce mois (jours / 7)
        return float(montant) * nb_jours / 7
    elif frequence == "Annuel":
        return float(montant) / 12
    return float(montant)

def injecter_recurrents(df, annee, mois_num):
    """Injecte les dépenses récurrentes actives pour le mois donné."""
    recs = charger_recurrents()
    if not recs: return df

    mois_p     = pd.Period(f"{annee}-{mois_num:02d}", freq="M")
    date_inj   = pd.Timestamp(f"{annee}-{mois_num:02d}-01")
    marker_col = "rec_id"

    if marker_col not in df.columns:
        df[marker_col] = ""

    deja_ids = set(df[df["date"].dt.to_period("M")==mois_p][marker_col].dropna())

    nouvelles = []
    a_desactiver = []

    for rid, r in recs.items():
        if rid in deja_ids: continue
        if not r.get("actif", True): continue

        # Vérifier que l'abonnement a commencé avant ou pendant ce mois
        debut = pd.Period(r.get("date_debut","2020-01"), freq="M")
        if mois_p < debut: continue

        mnt = montant_mensuel(r["montant"], r["frequence"], annee, mois_num)
        nouvelles.append({
            "date": date_inj, "categorie": r["categorie"],
            "montant": round(mnt, 2), "note": f"[AUTO] {r['nom']}",
            "photo": "", "rec_id": rid
        })

        # Si dernier_mois → désactiver après injection
        if r.get("dernier_mois", False):
            a_desactiver.append(rid)

    if nouvelles:
        df = pd.concat([df, pd.DataFrame(nouvelles)], ignore_index=True)
        sauvegarder_depenses(df)

    if a_desactiver:
        for rid in a_desactiver:
            recs[rid]["actif"] = False
            recs[rid]["dernier_mois"] = False
        sauvegarder_recurrents(recs)

    return df

# ── FONCTIONS : DÉPENSES ─────────────────────────────────────────────
def charger_depenses():
    if os.path.exists(FICHIER_DEPENSES):
        df = pd.read_csv(FICHIER_DEPENSES, dtype=str)
        for c in ["note","photo","rec_id"]:
            if c not in df.columns: df[c] = ""
        df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
        df["date"]    = pd.to_datetime(df["date"],    errors="coerce")
        df = df.dropna(subset=["date","montant"]).reset_index(drop=True)
        for c in ["note","photo","rec_id"]: df[c] = df[c].fillna("")
        return df
    return pd.DataFrame(columns=["date","categorie","montant","note","photo","rec_id"])

def sauvegarder_depenses(df):
    s = df.copy()
    s["date"] = s["date"].dt.strftime("%Y-%m-%d")
    for c in ["note","photo","rec_id"]:
        if c not in s.columns: s[c] = ""
    s.to_csv(FICHIER_DEPENSES, index=False)

def ajouter_depense(d, cat, mnt, note="", photo=""):
    df = charger_depenses()
    df = pd.concat([df, pd.DataFrame([{
        "date":pd.Timestamp(d),"categorie":cat,
        "montant":mnt,"note":note,"photo":photo,"rec_id":""
    }])], ignore_index=True)
    sauvegarder_depenses(df)

def supprimer_depense(idx):
    df = charger_depenses()
    if idx in df.index:
        p = df.at[idx,"photo"]
        if p and os.path.exists(p): os.remove(p)
        sauvegarder_depenses(df.drop(index=idx).reset_index(drop=True))

def sauvegarder_photo(f, d, cat):
    nom = f"{cat}_{d}_{f.name}".replace(" ","_")
    p   = DOSSIER_PHOTOS / nom
    p.write_bytes(f.getbuffer())
    return str(p)

def img_b64(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

# ── FONCTIONS : BUDGETS ──────────────────────────────────────────────
def charger_budgets():
    if os.path.exists(FICHIER_BUDGETS):
        df = pd.read_csv(FICHIER_BUDGETS, dtype=str)
        df["budget_mensuel"] = pd.to_numeric(df["budget_mensuel"], errors="coerce")
        return df
    return pd.DataFrame(columns=["categorie","budget_mensuel"])

def sauvegarder_budget(cat, val):
    df = charger_budgets()
    df = df[df["categorie"]!=cat]
    if val: df = pd.concat([df,pd.DataFrame([{"categorie":cat,"budget_mensuel":val}])],
                            ignore_index=True)
    df.to_csv(FICHIER_BUDGETS, index=False)

def budget_effectif(cat, df_bud, df_dep):
    r = df_bud[df_bud["categorie"]==cat]
    if not r.empty and pd.notna(r.iloc[0]["budget_mensuel"]):
        return float(r.iloc[0]["budget_mensuel"]), False
    h = df_dep[df_dep["categorie"]==cat].copy()
    if h.empty: return None, True
    h["m"] = h["date"].dt.to_period("M")
    moyenne = h.groupby("m")["montant"].sum().mean()
    if pd.isna(moyenne) or moyenne == 0: return None, True
    return float(moyenne), True

# ── PRÉDICTION ML ────────────────────────────────────────────────────
def predire(df, cat):
    h = df[df["categorie"]==cat].copy()
    if h.empty: return None,"aucune donnée"
    h["m"] = h["date"].dt.to_period("M")
    t = h.groupby("m")["montant"].sum().sort_index()
    if len(t)<4: return float(t.mean()),"moyenne"
    X = np.arange(len(t)).reshape(-1,1)
    p = max(0, LinearRegression().fit(X,t.values).predict([[len(t)]])[0])
    return float(p),"régression linéaire"

# ── MODALE IMAGE ─────────────────────────────────────────────────────
@st.dialog("Aperçu du reçu")
def afficher_image_modale(path):
    ext = Path(path).suffix.lower()
    if ext in [".png",".jpg",".jpeg",".webp"]:
        st.image(path, use_container_width=True)
    else:
        st.info("Fichier joint (non prévisualisable)")

# ── CHARGEMENT ────────────────────────────────────────────────────────
for k in ["popup","img_modal"]:
    if k not in st.session_state: st.session_state[k] = None

categories, icones = get_categories()
now      = pd.Timestamp.now()
annee_m  = now.year
mois_num = now.month
mois_p   = now.to_period("M")

df  = charger_depenses()
df  = injecter_recurrents(df, annee_m, mois_num)
bud = charger_budgets()
recs = charger_recurrents()
dm  = df[df["date"].dt.to_period("M")==mois_p]

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("logo.svg"):
        st.image("logo.svg", use_container_width=True)
    st.markdown('<div class="logo-sub">B U D G E T // S Y S</div>'
                '<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.metric("Total ce mois", f"{dm['montant'].sum():.2f} €")

    recs_actifs = {n:v for n,v in recs.items() if v.get("actif",True)}
    if recs_actifs:
        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
        st.markdown("**Récurrents actifs**")
        for rid,r in recs_actifs.items():
            mnt_m = montant_mensuel(r["montant"],r["frequence"],annee_m,mois_num)
            lbl   = "⚠ Dernier mois" if r.get("dernier_mois") else r["frequence"]
            st.markdown(f"{icones.get(r['categorie'],ICONES_CUSTOM_DEFAULT)} "
                        f"**{r['nom']}** — {mnt_m:.2f} €/mois *({lbl})*")

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.caption("⚙ 100% local · aucune donnée transmise")

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown('<p class="scan">SYS::BUDGET // SESSION ACTIVE // DONNÉES LOCALES</p>',
            unsafe_allow_html=True)
st.title("BUDGET//SYS")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Dépenses ce mois", f"{dm['montant'].sum():.2f} €")
c2.metric("Dépenses ponctuelles", len(dm[dm["rec_id"]==""]))
c3.metric("Dépenses récurrentes", len(dm[dm["rec_id"]!=""]))
c4.metric("Mois d'historique",
          df["date"].dt.to_period("M").nunique() if not df.empty else 0)
st.markdown("---")

# ── INVENTAIRE ────────────────────────────────────────────────────────
EMOJI_SUGGESTIONS = [
    "🏡","🚗","🚌","✈️","🍕","🛒","☕","🎬","🎵","🏋️",
    "💊","👗","👟","📱","💻","🐶","🌱","⚽","🎓","🏖️",
    "🔧","💡","🎁","🍺","🧴","📚","🎮","🧳","🏦","💰",
]

if "show_add_cat" not in st.session_state:
    st.session_state.show_add_cat = False

col_inv, col_add = st.columns([5,1])
with col_inv:
    st.subheader("▸ INVENTAIRE")
with col_add:
    st.markdown("<br>", unsafe_allow_html=True)
    lbl_btn = "－ Catégorie" if st.session_state.show_add_cat else "＋ Catégorie"
    if st.button(lbl_btn, use_container_width=True):
        st.session_state.show_add_cat = not st.session_state.show_add_cat
        st.rerun()

if st.session_state.show_add_cat:
    with st.container(border=True):
        st.markdown("**Nouvelle catégorie**")
        ca, cb = st.columns([3,3])
        with ca:
            nom_nc = st.text_input("Nom", key="nc_nom", placeholder="Ex: Vacances")
        with cb:
            st.markdown("Icône — choisissez ou saisissez")
            emoji_cols = st.columns(10)
            selected_emoji = st.session_state.get("nc_ico_sel", "🔷")
            for ei, em in enumerate(EMOJI_SUGGESTIONS):
                with emoji_cols[ei % 10]:
                    if st.button(em, key=f"em_{ei}"):
                        st.session_state.nc_ico_sel = em
                        st.rerun()
            ico_nc = st.text_input("Ou saisir manuellement",
                value=st.session_state.get("nc_ico_sel","🔷"), key="nc_ico_txt")

        col_sup, col_add2 = st.columns([3,1])
        with col_sup:
            st.caption("Catégories personnalisées existantes :")
            cat_data_cur = charger_categories()
            for c in cat_data_cur["categories"]:
                if c not in CATEGORIES_DEFAULT:
                    cc1,cc2 = st.columns([4,1])
                    with cc1: st.markdown(f"{cat_data_cur['icones'].get(c,'🔷')} {c}")
                    with cc2:
                        if st.button("🗑", key=f"del_cat_{c}",
                                     help="Supprimer cette catégorie"):
                            st.session_state[f"confirm_del_cat_{c}"] = True
                            st.rerun()
                    if st.session_state.get(f"confirm_del_cat_{c}"):
                        st.warning(f"Supprimer « {c} » ?")
                        oc1,oc2 = st.columns(2)
                        with oc1:
                            if st.button("✓ Oui", key=f"yes_cat_{c}"):
                                cat_data_cur["categories"].remove(c)
                                cat_data_cur["icones"].pop(c,None)
                                sauvegarder_categories(cat_data_cur)
                                del st.session_state[f"confirm_del_cat_{c}"]
                                st.rerun()
                        with oc2:
                            if st.button("✗ Non", key=f"no_cat_{c}"):
                                del st.session_state[f"confirm_del_cat_{c}"]
                                st.rerun()
        with col_add2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("＋ Ajouter", key="nc_ok", use_container_width=True):
                if nom_nc.strip() and nom_nc.strip() not in categories:
                    cat_data = charger_categories()
                    cat_data["categories"].append(nom_nc.strip())
                    cat_data["icones"][nom_nc.strip()] = ico_nc or ICONES_CUSTOM_DEFAULT
                    sauvegarder_categories(cat_data)
                    st.session_state.show_add_cat = False
                    st.session_state.pop("nc_ico_sel", None)
                    st.rerun()

cols = st.columns(4)
for i, cat in enumerate(categories):
    with cols[i%4]:
        x    = dm[dm["categorie"]==cat]["montant"].sum()
        z, a = budget_effectif(cat, bud, df)
        ratio= (x/z) if z and z>0 else 0

        if   ratio>1.0:  couleur,classe = ROUGE,"alert"
        elif ratio>=0.8: couleur,classe = OR,   "warn"
        else:            couleur,classe = CYAN,  "ok"

        z_str = f"{z:.0f}" if z else "∞"
        pct   = min(int(ratio*100),100)
        rec_cat = [r for r in recs_actifs.values() if r["categorie"]==cat]
        badge = ""
        if rec_cat:
            badge = f'<span class="rec-badge">{len(rec_cat)} AUTO</span>'

        st.markdown(f"""
<div class="inv-card {classe}">
  <div class="inv-icone">{icones.get(cat,ICONES_CUSTOM_DEFAULT)}</div>
  <div class="inv-nom">{cat} {badge}</div>
  <div class="inv-xz">{x:.0f} € / {z_str} €</div>
  <div class="inv-bar-bg">
    <div class="inv-bar" style="width:{pct}%;background:{couleur};"></div>
  </div>
</div>""", unsafe_allow_html=True)

        if st.button(f"⚙ {cat}", key=f"b_{cat}", use_container_width=True):
            st.session_state.popup = cat if st.session_state.popup!=cat else None
            st.rerun()

st.markdown("---")

# ── RÉCAPITULATIF RÉCURRENTS ──────────────────────────────────────────
if recs:
    with st.expander("📋 RÉCAPITULATIF — CHARGES RÉCURRENTES", expanded=True):
        total_rec_mois = 0
        lignes_rec = []
        for rid, r in sorted(recs.items(), key=lambda x: x[1]["categorie"]):
            actif   = r.get("actif", True)
            mnt_m   = montant_mensuel(r["montant"], r["frequence"], annee_m, mois_num)
            statut  = "🟢 Actif" if actif and not r.get("dernier_mois") else                       ("⚠️ Dernier mois" if r.get("dernier_mois") else "⛔ Arrêté")
            if actif: total_rec_mois += mnt_m
            lignes_rec.append({
                "Catégorie":  r["categorie"],
                "Nom":        r["nom"],
                "Montant":    f"{r['montant']:.2f} €",
                "Fréquence":  r["frequence"],
                "≈ /mois":    f"{mnt_m:.2f} €",
                "Début":      r.get("date_debut","?"),
                "Statut":     statut,
            })
        st.dataframe(pd.DataFrame(lignes_rec), use_container_width=True, hide_index=True)
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("Total récurrents actifs / mois", f"{total_rec_mois:.2f} €")
        c_r2.metric("Nombre de récurrents", len([r for r in recs.values() if r.get("actif",True)]))

st.markdown("---")

# ── PANNEAU CATÉGORIE ─────────────────────────────────────────────────
if st.session_state.popup:
    cat = st.session_state.popup
    st.subheader(f"{icones.get(cat,ICONES_CUSTOM_DEFAULT)}  {cat.upper()}")

    tabs = st.tabs(["➕ Ponctuelle","🔄 Récurrents","🗑 Dépenses","💰 Budget"])

    # ── TAB 0 : Ajouter dépense ponctuelle ───────────────────────────
    with tabs[0]:
        with st.form(f"f_add_{cat}", clear_on_submit=True):
            ca,cb = st.columns(2)
            with ca:
                d_i = st.date_input("Date", value=date.today())
                m_i = st.number_input("Montant (€)", min_value=0.01,
                                       step=0.5, value=10.0)
            with cb:
                n_i = st.text_input("Note", placeholder="ex : courses Lidl")
                p_i = st.file_uploader("Photo / reçu",
                        type=["png","jpg","jpeg","webp"], key=f"pu_{cat}")
            ok = st.form_submit_button("⏎ Enregistrer")
        if ok:
            ph = sauvegarder_photo(p_i,d_i,cat) if p_i else ""
            ajouter_depense(d_i, cat, m_i, n_i, ph)
            st.success(f"✓  {m_i:.2f} € ajouté en {cat}")
            st.rerun()

    # ── TAB 1 : Récurrents de cette catégorie ────────────────────────
    with tabs[1]:
        rec_cat = {rid:r for rid,r in recs.items() if r["categorie"]==cat}

        if rec_cat:
            for rid, r in list(rec_cat.items()):
                actif        = r.get("actif",True)
                dernier_mois = r.get("dernier_mois",False)
                mnt_m = montant_mensuel(r["montant"],r["frequence"],annee_m,mois_num)

                ca,cb,cc,cc2,cd,ce = st.columns([3,2,2,1.5,1,1])
                with ca:
                    st.markdown(f"**{r['nom']}**")
                    st.caption(f"Depuis {r.get('date_debut','?')} · {r['frequence']}")
                with cb:
                    st.markdown(f"{r['montant']:.2f} €/{r['frequence'].lower()}")
                    st.caption(f"≈ {mnt_m:.2f} €/mois")
                with cc:
                    if not actif:
                        st.markdown("⛔ Arrêté")
                    elif dernier_mois:
                        st.markdown("⚠️ Dernier mois")
                    else:
                        st.markdown("🟢 Actif")
                with cc2:
                    ph_r = str(r.get("photo",""))
                    if ph_r and os.path.exists(ph_r):
                        if st.button("🔍 Doc", key=f"vr_{rid}"):
                            afficher_image_modale(ph_r)
                with cd:
                    if actif and not dernier_mois:
                        if st.button("⚠", key=f"dm_{rid}",
                                     help="Marquer comme dernier mois"):
                            recs[rid]["dernier_mois"] = True
                            sauvegarder_recurrents(recs)
                            st.rerun()
                    elif actif and dernier_mois:
                        if st.button("↩", key=f"ann_{rid}",
                                     help="Annuler — continuer l'abonnement"):
                            recs[rid]["dernier_mois"] = False
                            sauvegarder_recurrents(recs)
                            st.rerun()
                    else:
                        if st.button("▶", key=f"react_{rid}",
                                     help="Réactiver"):
                            recs[rid]["actif"] = True
                            sauvegarder_recurrents(recs)
                            st.rerun()
                with ce:
                    if st.button("🗑", key=f"del_rec_{rid}",
                                 help="Supprimer définitivement"):
                        st.session_state[f"confirm_del_rec_{rid}"] = True
                        st.rerun()
                if st.session_state.get(f"confirm_del_rec_{rid}"):
                    st.warning(f"Supprimer « {r['nom']} » définitivement ?")
                    c1c,c2c = st.columns(2)
                    with c1c:
                        if st.button("✓ Oui, supprimer", key=f"yes_rec_{rid}"):
                            del recs[rid]
                            sauvegarder_recurrents(recs)
                            st.session_state.pop(f"confirm_del_rec_{rid}", None)
                            st.rerun()
                    with c2c:
                        if st.button("✗ Annuler", key=f"no_rec_{rid}"):
                            st.session_state.pop(f"confirm_del_rec_{rid}", None)
                            st.rerun()
                st.divider()
        else:
            st.info("Aucun récurrent pour cette catégorie.")

        st.markdown("**Ajouter un récurrent**")
        with st.form(f"f_rec_{cat}", clear_on_submit=True):
            ca,cb = st.columns(2)
            with ca:
                nom_r  = st.text_input("Nom", placeholder="Loyer, Netflix…")
                mnt_r  = st.number_input("Montant (€)", min_value=0.01,
                                          step=0.5, value=10.0)
            with cb:
                freq_r = st.selectbox("Fréquence", FREQUENCES)
                deb_r  = st.date_input("Date de début", value=date.today())
            photo_r = st.file_uploader("Photo / document (contrat, confirmation…)",
                        type=["png","jpg","jpeg","webp"], key=f"pr_{cat}")
            ok_r = st.form_submit_button("⏎ Ajouter le récurrent")
        if ok_r and nom_r.strip():
            import uuid
            rid_new = str(uuid.uuid4())[:8]
            ph_r = sauvegarder_photo(photo_r, deb_r, f"rec_{cat}") if photo_r else ""
            recs[rid_new] = {
                "nom": nom_r.strip(), "categorie": cat,
                "montant": float(mnt_r), "frequence": freq_r,
                "date_debut": deb_r.strftime("%Y-%m"),
                "actif": True, "dernier_mois": False,
                "photo": ph_r
            }
            sauvegarder_recurrents(recs)
            st.success(f"✓ « {nom_r} » ajouté — {mnt_r:.2f} €/{freq_r.lower()}")
            st.rerun()

    # ── TAB 2 : Gérer / supprimer dépenses ───────────────────────────
    with tabs[2]:
        dep_cat = df[df["categorie"]==cat].sort_values("date",ascending=False)
        if dep_cat.empty:
            st.info("Aucune dépense pour cette catégorie.")
        else:
            for idx, row in dep_cat.iterrows():
                is_auto = bool(row.get("rec_id",""))
                ca,cb,cc,cd = st.columns([2,4,2,1])
                with ca:
                    st.code(row["date"].strftime("%d/%m/%Y"), language=None)
                with cb:
                    note = f" · *{row['note']}*" if row.get("note") else ""
                    tag  = " 🔄" if is_auto else ""
                    st.markdown(f"**{row['montant']:.2f} €**{tag}{note}")
                with cc:
                    ph = str(row.get("photo",""))
                    if ph and os.path.exists(ph):
                        ext = Path(ph).suffix.lower()
                        if ext in [".png",".jpg",".jpeg",".webp"]:
                            b64  = img_b64(ph)
                            mime = "image/jpeg" if ext in [".jpg",".jpeg"] \
                                   else f"image/{ext[1:]}"
                            if st.button("🔍 Voir reçu", key=f"view_{idx}"):
                                afficher_image_modale(ph)
                with cd:
                    if st.button("🗑", key=f"sd_{idx}", help="Supprimer"):
                        st.session_state[f"confirm_dep_{idx}"] = True
                        st.rerun()
                if st.session_state.get(f"confirm_dep_{idx}"):
                    st.warning("Supprimer cette dépense ?")
                    c1d,c2d = st.columns(2)
                    with c1d:
                        if st.button("✓ Oui", key=f"yes_dep_{idx}"):
                            supprimer_depense(idx)
                            st.session_state.pop(f"confirm_dep_{idx}", None)
                            st.rerun()
                    with c2d:
                        if st.button("✗ Non", key=f"no_dep_{idx}"):
                            st.session_state.pop(f"confirm_dep_{idx}", None)
                            st.rerun()
                st.divider()

    # ── TAB 3 : Budget Z ─────────────────────────────────────────────
    with tabs[3]:
        z_cur, auto_cur = budget_effectif(cat, bud, df)
        if z_cur and not auto_cur:
            st.info(f"Budget actuel : **{z_cur:.2f} €/mois**")
        else:
            st.caption("Pas de budget fixe — moyenne historique utilisée.")
        with st.form(f"f_bud_{cat}", clear_on_submit=False):
            nv = st.number_input("Nouveau budget Z (0 = retour auto)",
                                  min_value=0.0, step=10.0,
                                  value=float(z_cur or 0))
            ok2 = st.form_submit_button("⏎ Définir")
        if ok2:
            sauvegarder_budget(cat, nv if nv>0 else None)
            st.success(f"Budget {cat} → {nv:.0f} €/mois")
            st.rerun()

    if st.button("✕ Fermer", key="close_popup"):
        st.session_state.popup = None
        st.rerun()
    st.markdown("---")

# ── ALERTES ───────────────────────────────────────────────────────────
st.subheader("⚠ ALERTES")
alert = False
for cat in categories:
    x = dm[dm["categorie"]==cat]["montant"].sum()
    if x==0: continue
    z,a = budget_effectif(cat,bud,df)
    if not z: continue
    r   = x/z if z>0 else 0
    src = "moy. auto" if a else "budget fixé"
    if r>1.0:
        st.error(f"🔴 **{cat}** — {x:.2f} € / {z:.2f} € ({src}) "
                 f"— dépassement +{(r-1)*100:.0f}%")
        alert = True
    elif r>=0.8:
        st.warning(f"🟡 **{cat}** — {x:.2f} € / {z:.2f} € ({src}) "
                   f"— {r*100:.0f}% du seuil")
        alert = True
    elif r==1.0:
        st.success(f"🎯 **{cat}** — objectif atteint exactement ({x:.2f} €)")
if not alert:
    st.success("🟢 Aucun seuil franchi ce mois-ci.")
st.markdown("---")

# ── GRAPHIQUES ────────────────────────────────────────────────────────
if not df.empty:
    plt.rcParams.update({
        "figure.facecolor":"#0E0E1A","axes.facecolor":"#0E0E1A",
        "axes.edgecolor":"#333355","axes.labelcolor":"#AAAACC",
        "text.color":"#CCCCEE","xtick.color":"#666688","ytick.color":"#666688",
    })
    cg,cd = st.columns(2)
    with cg:
        st.subheader("▸ ÉVOLUTION MENSUELLE")
        df2 = df.copy()
        df2["mois"] = df2["date"].dt.to_period("M").astype(str)
        tot = df2.groupby("mois")["montant"].sum()
        fig,ax = plt.subplots(figsize=(6,3.5))
        tot.plot(kind="bar",ax=ax,color=VIOLET,width=.65)
        ax.set_ylabel("€"); ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=40,ha="right",fontsize=8)
        plt.tight_layout(); st.pyplot(fig)
    with cd:
        st.subheader("▸ RÉPARTITION")
        tot_c = df.groupby("categorie")["montant"].sum().sort_values(ascending=False)
        clrs  = [VIOLET,CYAN,VERT,"#FF8C42",ROUGE,"#8888FF",OR,"#666688","#AA88FF","#44CCAA"]
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.pie(tot_c.values, labels=tot_c.index, autopct="%1.0f%%",
               colors=clrs[:len(tot_c)],
               textprops={"color":"#DDDDEE","fontsize":8},
               pctdistance=.82, startangle=90)
        plt.tight_layout(); st.pyplot(fig)
    st.markdown("---")

# ── PROJECTION ML ─────────────────────────────────────────────────────
st.subheader("▸ PROJECTION // MOIS SUIVANT")
rows = []
for cat in categories:
    if df.empty or df[df["categorie"]==cat].empty: continue
    p,meth = predire(df,cat)
    z,_    = budget_effectif(cat,bud,df)
    rows.append({"Catégorie":cat,"Projection (€)":f"{p:.2f}",
                 "Méthode":meth,"Ref. budget (€)":f"{z:.2f}" if z else "—"})
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("Pas assez de données.")
st.markdown("---")

# ── AGENT IA ──────────────────────────────────────────────────────────
st.subheader("🤖 AGENT")
q = st.text_input("Question en langage naturel",
    placeholder="Ex : Combien en Alimentation ce mois ?",key="q_agent")
if q:
    st.markdown(f"> {agent_repondre(q,df)}")
with st.expander("Exemples"):
    st.markdown("""
- « Combien j'ai dépensé en Alimentation ? »
- « Ma catégorie la plus chère ? »
- « Moyenne Transport »
""")
st.markdown("---")

# ── HISTORIQUE ────────────────────────────────────────────────────────
st.subheader("▸ HISTORIQUE")
if not df.empty:
    aff = df[["date","categorie","montant","note","rec_id"]].sort_values(
        "date",ascending=False).copy()
    aff["date"] = aff["date"].dt.strftime("%d/%m/%Y")
    aff["type"] = aff["rec_id"].apply(lambda x: "🔄 Auto" if x else "✏ Ponctuel")
    st.dataframe(aff[["date","categorie","type","montant","note"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("Aucune dépense pour l'instant.")
st.markdown("---")



# ── IMPORT CSV ────────────────────────────────────────────────────────
with st.expander("▸ Import CSV"):
    st.caption("Colonnes : date, categorie, montant (note optionnel)")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up:
        di = pd.read_csv(up, dtype=str)
        di["montant"] = pd.to_numeric(di["montant"], errors="coerce")
        di["date"]    = pd.to_datetime(di["date"],    errors="coerce")
        di = di.dropna(subset=["date","montant"])
        sauvegarder_depenses(
            pd.concat([charger_depenses(),di],ignore_index=True).drop_duplicates())
        st.success(f"✓ {len(di)} entrées importées.")
        st.rerun()
