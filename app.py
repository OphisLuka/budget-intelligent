"""
Budget personnel intelligent — v4
Thème : hologramme Silver Wolf / Honkai Star Rail
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json, base64
from datetime import date
from pathlib import Path
from sklearn.linear_model import LinearRegression
from agent import repondre as agent_repondre

# ── CONSTANTES ────────────────────────────────────────────────────────
FICHIER_DEPENSES    = "depenses.csv"
FICHIER_BUDGETS     = "budgets.csv"
FICHIER_ABONNEMENTS = "abonnements.json"
DOSSIER_PHOTOS      = Path("photos_depenses")
DOSSIER_PHOTOS.mkdir(exist_ok=True)

CATEGORIES_RESET        = ["Logement", "Alimentation", "Transport",
                            "Loisirs", "Santé", "Shopping", "Autre"]
CATEGORIES_PERSISTANTES = ["Abonnements"]
CATEGORIES              = CATEGORIES_RESET + CATEGORIES_PERSISTANTES

ICONES = {
    "Logement":"🏠","Alimentation":"🍽️","Transport":"🚇",
    "Loisirs":"🎮","Santé":"💊","Shopping":"🛍️",
    "Autre":"📦","Abonnements":"📡",
}

VIOLET = "#A855F7"; CYAN = "#00E5FF"; VERT = "#39FF6A"
ROUGE  = "#FF3860"; OR   = "#FFD700"; FOND = "#0A0A12"

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(page_title="BUDGET//SYS", layout="wide", page_icon="🟣")

# ── LOGO SVG hologramme (fidèle à l'image référence) ─────────────────
LOGO_SVG = """
<svg width="120" height="152" viewBox="0 0 120 152"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lg_fill" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#7EC8E3" stop-opacity="0.22"/>
      <stop offset="50%"  stop-color="#B8D8F0" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#6AAED6" stop-opacity="0.15"/>
    </linearGradient>
    <linearGradient id="lg_shine" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.28"/>
      <stop offset="45%"  stop-color="#ffffff" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="lg_row" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#FFFFFF" stop-opacity="0.0"/>
      <stop offset="8%"   stop-color="#FFFFFF" stop-opacity="0.85"/>
      <stop offset="92%"  stop-color="#FFFFFF" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0.0"/>
    </linearGradient>
    <clipPath id="card_clip">
      <rect x="2" y="2" width="116" height="148" rx="6"/>
    </clipPath>
  </defs>

  <!-- Ombre portée douce -->
  <rect x="4" y="5" width="116" height="148" rx="7"
        fill="#000" opacity="0.35"/>

  <!-- Corps principal : fond bleu translucide -->
  <rect x="2" y="2" width="116" height="148" rx="6"
        fill="url(#lg_fill)" stroke="#0A0A18" stroke-width="2.5"/>

  <!-- Reflet haut -->
  <rect x="2" y="2" width="116" height="72" rx="6"
        fill="url(#lg_shine)" clip-path="url(#card_clip)"/>

  <!-- Bord lumineux haut (trait cyan) -->
  <line x1="10" y1="2.5" x2="110" y2="2.5"
        stroke="#00E5FF" stroke-width="1.2" opacity="0.7"/>

  <!-- Coins lumineux -->
  <rect x="2"   y="2"   width="6" height="6" rx="0"
        fill="none" stroke="#00E5FF" stroke-width="1.2" opacity="0.9"/>
  <rect x="112" y="2"   width="6" height="6" rx="0"
        fill="none" stroke="#00E5FF" stroke-width="1.2" opacity="0.9"/>
  <rect x="2"   y="144" width="6" height="6" rx="0"
        fill="none" stroke="#A855F7" stroke-width="1.2" opacity="0.7"/>
  <rect x="112" y="144" width="6" height="6" rx="0"
        fill="none" stroke="#A855F7" stroke-width="1.2" opacity="0.7"/>

  <!-- TITRE en haut -->
  <rect x="14" y="14" width="92" height="7" rx="2"
        fill="url(#lg_row)" opacity="0.9"/>
  <rect x="14" y="24" width="60" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.5"/>

  <!-- Séparateur -->
  <line x1="14" y1="35" x2="106" y2="35"
        stroke="#FFFFFF" stroke-width="0.5" opacity="0.18"/>

  <!-- Ligne 1 : icône carré + texte -->
  <rect x="14" y="41" width="9" height="9" rx="1"
        fill="#FFFFFF" opacity="0.55"/>
  <rect x="28" y="43" width="52" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.8"/>
  <rect x="84" y="43" width="22" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.45"/>

  <!-- Ligne 2 -->
  <rect x="14" y="55" width="9" height="9" rx="1"
        fill="#FFFFFF" opacity="0.45"/>
  <rect x="28" y="57" width="44" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.7"/>
  <rect x="76" y="57" width="30" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.35"/>

  <!-- Ligne 3 -->
  <rect x="14" y="69" width="9" height="9" rx="1"
        fill="#FFFFFF" opacity="0.38"/>
  <rect x="28" y="71" width="58" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.65"/>
  <rect x="90" y="71" width="16" height="4" rx="2"
        fill="url(#lg_row)" opacity="0.3"/>

  <!-- Séparateur -->
  <line x1="14" y1="84" x2="106" y2="84"
        stroke="#FFFFFF" stroke-width="0.5" opacity="0.15"/>

  <!-- Mini graphique barres -->
  <rect x="14" y="101" width="10" height="19" rx="1" fill="#00E5FF" opacity="0.55"/>
  <rect x="27" y="108" width="10" height="12" rx="1" fill="#A855F7" opacity="0.55"/>
  <rect x="40" y="96"  width="10" height="24" rx="1" fill="#00E5FF" opacity="0.40"/>
  <rect x="53" y="104" width="10" height="16" rx="1" fill="#A855F7" opacity="0.40"/>
  <rect x="66" y="110" width="10" height="10" rx="1" fill="#00E5FF" opacity="0.30"/>
  <rect x="79" y="99"  width="10" height="21" rx="1" fill="#A855F7" opacity="0.30"/>
  <rect x="92" y="105" width="10" height="15" rx="1" fill="#00E5FF" opacity="0.25"/>
  <line x1="14" y1="121" x2="106" y2="121"
        stroke="#FFFFFF" stroke-width="0.4" opacity="0.2"/>

  <!-- Ligne statut bas -->
  <rect x="14" y="130" width="14" height="3" rx="1" fill="#39FF6A" opacity="0.7"/>
  <rect x="32" y="130" width="34" height="3" rx="1" fill="#FFFFFF"  opacity="0.25"/>
  <rect x="70" y="130" width="20" height="3" rx="1" fill="#FFFFFF"  opacity="0.15"/>
  <rect x="94" y="130" width="12" height="3" rx="1" fill="#A855F7"  opacity="0.4"/>

  <!-- Ligne 4 bas -->
  <rect x="14" y="138" width="92" height="3" rx="1"
        fill="url(#lg_row)" opacity="0.3"/>
</svg>
"""

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap');

.stApp {{
    background:{FOND};
    background-image:
        radial-gradient(circle at 12% 8%,  rgba(168,85,247,.09) 0%,transparent 38%),
        radial-gradient(circle at 88% 92%, rgba(0,229,255,.06)  0%,transparent 38%);
}}
html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;color:#E0E0F0;}}
h1,h2,h3{{font-family:'JetBrains Mono',monospace!important;letter-spacing:.02em;}}
h1{{color:{VIOLET}!important;text-shadow:0 0 18px rgba(168,85,247,.4);
    border-bottom:1px solid rgba(168,85,247,.25);padding-bottom:.4em;}}
h2,h3{{color:{CYAN}!important;}}

/* ── Inventaire ── */
.inv-card{{
    background:linear-gradient(145deg,rgba(16,16,28,.95),rgba(24,14,42,.9));
    border:1px solid rgba(168,85,247,.3);border-radius:6px;
    padding:14px 12px 11px;text-align:center;position:relative;overflow:hidden;
    margin-bottom:4px;transition:border-color .18s,box-shadow .18s;
}}
.inv-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,229,255,.55),transparent);}}
.inv-card.ok      {{border-color:rgba(168,85,247,.3);}}
.inv-card.warning {{border-color:{OR};  box-shadow:0 0 12px rgba(255,215,0,.2);}}
.inv-card.alerte  {{border-color:{ROUGE};box-shadow:0 0 12px rgba(255,56,96,.25);}}
.inv-card.persist {{border-color:rgba(0,229,255,.5);}}
.inv-icone{{font-size:1.9em;line-height:1.2;}}
.inv-nom{{font-family:'JetBrains Mono',monospace;font-size:.68em;
    letter-spacing:.09em;color:#7777AA;text-transform:uppercase;margin:5px 0 7px;}}
.inv-xz{{font-family:'JetBrains Mono',monospace;font-size:1em;
    font-weight:600;color:{CYAN};}}
.inv-bar-bg{{background:rgba(255,255,255,.07);border-radius:3px;
    height:4px;margin-top:9px;overflow:hidden;}}
.inv-bar{{height:100%;border-radius:3px;}}

/* ── Sidebar ── */
[data-testid="stSidebar"]{{
    background:#0B0B17;
    border-right:1px solid rgba(0,229,255,.15);
}}
.logo-wrap{{display:flex;flex-direction:column;align-items:center;
    padding:8px 0 4px;gap:6px;}}
.logo-sub{{font-family:'JetBrains Mono',monospace;font-size:.6em;
    color:rgba(0,229,255,.45);letter-spacing:.2em;text-align:center;}}
.sidebar-sep{{height:1px;background:linear-gradient(90deg,
    transparent,rgba(0,229,255,.2),transparent);margin:8px 0;}}

/* ── Boutons ── */
.stButton button,.stFormSubmitButton button{{
    background:transparent;border:1px solid {VIOLET};color:{VIOLET};
    font-family:'JetBrains Mono',monospace;text-transform:uppercase;
    letter-spacing:.05em;border-radius:2px;transition:all .15s;
}}
.stButton button:hover,.stFormSubmitButton button:hover{{
    background:{VIOLET};color:{FOND};box-shadow:0 0 14px rgba(168,85,247,.55);
}}

/* ── Métriques ── */
[data-testid="stMetricValue"]{{
    font-family:'JetBrains Mono',monospace!important;
    color:{CYAN}!important;text-shadow:0 0 8px rgba(0,229,255,.35);
}}
[data-testid="stMetricLabel"]{{
    font-family:'JetBrains Mono',monospace!important;
    font-size:.72em!important;letter-spacing:.08em;
    text-transform:uppercase;color:#6666AA!important;
}}

/* ── Inputs ── */
input,textarea,select{{font-family:'JetBrains Mono',monospace!important;}}
[data-testid="stDataFrame"]{{border:1px solid rgba(168,85,247,.18);}}

/* ── Utilitaires ── */
.scan{{font-family:'JetBrains Mono',monospace;font-size:.7em;
    letter-spacing:.14em;color:rgba(0,229,255,.4);
    border-bottom:1px dashed rgba(0,229,255,.18);padding-bottom:.5em;}}
.badge-auto{{font-family:'JetBrains Mono',monospace;font-size:.55em;
    color:{CYAN};border:1px solid rgba(0,229,255,.4);
    border-radius:2px;padding:1px 4px;}}
.photo-thumb{{border-radius:3px;border:1px solid rgba(0,229,255,.25);
    max-width:76px;max-height:56px;object-fit:cover;}}
.section-title{{font-family:'JetBrains Mono',monospace;font-size:.78em;
    color:rgba(0,229,255,.55);letter-spacing:.1em;text-transform:uppercase;
    margin:0 0 6px;}}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("logo.svg"):
        st.sidebar.image("logo.svg", use_container_width=True)
    st.markdown('<div class="logo-sub">B U D G E T // S Y S</div>'
                '<div class="sidebar-sep"></div>', unsafe_allow_html=True)

# ── FONCTIONS : DÉPENSES ─────────────────────────────────────────────
def charger_depenses():
    if os.path.exists(FICHIER_DEPENSES):
        df = pd.read_csv(FICHIER_DEPENSES, dtype=str)
        for c in ["note","photo"]:
            if c not in df.columns: df[c] = ""
        df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
        df["date"]    = pd.to_datetime(df["date"],    errors="coerce")
        df = df.dropna(subset=["date","montant"]).reset_index(drop=True)
        df["note"]  = df["note"].fillna("")
        df["photo"] = df["photo"].fillna("")
        return df
    return pd.DataFrame(columns=["date","categorie","montant","note","photo"])

def sauvegarder_depenses(df):
    s = df.copy()
    s["date"] = s["date"].dt.strftime("%Y-%m-%d")
    for c in ["note","photo"]:
        if c not in s.columns: s[c] = ""
    s.to_csv(FICHIER_DEPENSES, index=False)

def ajouter_depense(d, cat, mnt, note="", photo=""):
    df = charger_depenses()
    df = pd.concat([df, pd.DataFrame([{
        "date":pd.Timestamp(d),"categorie":cat,
        "montant":mnt,"note":note,"photo":photo
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
    df = df[df["categorie"] != cat]
    if val: df = pd.concat([df, pd.DataFrame([{"categorie":cat,"budget_mensuel":val}])],
                            ignore_index=True)
    df.to_csv(FICHIER_BUDGETS, index=False)

def budget_effectif(cat, df_bud, df_dep):
    r = df_bud[df_bud["categorie"]==cat]
    if not r.empty and pd.notna(r.iloc[0]["budget_mensuel"]):
        return float(r.iloc[0]["budget_mensuel"]), False
    h = df_dep[df_dep["categorie"]==cat].copy()
    if h.empty: return None, True
    h["m"] = h["date"].dt.to_period("M")
    return float(h.groupby("m")["montant"].sum().mean()), True

# ── FONCTIONS : ABONNEMENTS ──────────────────────────────────────────
def charger_abonnements():
    if not os.path.exists(FICHIER_ABONNEMENTS): return {}
    with open(FICHIER_ABONNEMENTS,"r",encoding="utf-8") as f:
        data = json.load(f)
    for k,v in data.items():
        if isinstance(v,(int,float)): data[k]={"montant":v,"actif":True}
    return data

def sauvegarder_abonnements(d):
    with open(FICHIER_ABONNEMENTS,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False,indent=2)

def injecter_abonnements(df):
    abo    = charger_abonnements()
    actifs = {n:v for n,v in abo.items() if v.get("actif",True)}
    if not actifs: return df
    mois       = pd.Timestamp.now().to_period("M")
    date_inj   = pd.Timestamp(mois.start_time.date())
    deja       = df[(df["categorie"]=="Abonnements") &
                    (df["date"].dt.to_period("M")==mois) &
                    (df["date"]==date_inj)]
    if deja.empty:
        rows = [{"date":date_inj,"categorie":"Abonnements",
                 "montant":float(v["montant"]),"note":n,"photo":""}
                for n,v in actifs.items()]
        df = pd.concat([df,pd.DataFrame(rows)], ignore_index=True)
        sauvegarder_depenses(df)
    return df

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

# ── CHARGEMENT ────────────────────────────────────────────────────────
if "popup" not in st.session_state: st.session_state.popup = None

df   = charger_depenses()
df   = injecter_abonnements(df)
bud  = charger_budgets()
abo  = charger_abonnements()
mois = pd.Timestamp.now().to_period("M")
dm   = df[df["date"].dt.to_period("M")==mois]

# ── SIDEBAR RÉSUMÉ ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">Résumé du mois</p>', unsafe_allow_html=True)
    st.metric("Total dépensé", f"{dm['montant'].sum():.2f} €")
    actifs_abo = {n:v for n,v in abo.items() if v.get("actif",True)}
    if actifs_abo:
        st.markdown('<div class="sidebar-sep"></div>'
                    '<p class="section-title">Abonnements actifs</p>',
                    unsafe_allow_html=True)
        for nom,val in actifs_abo.items():
            st.markdown(f"📡 **{nom}** — {val['montant']:.2f} €/mois")
        st.markdown(f"**Total : {sum(v['montant'] for v in actifs_abo.values()):.2f} €/mois**")
    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.caption("⚙ Données 100% locales")

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown('<p class="scan">SYS::BUDGET // SESSION ACTIVE // DONNÉES LOCALES</p>',
            unsafe_allow_html=True)
st.title("BUDGET//SYS")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Dépenses ce mois", f"{dm['montant'].sum():.2f} €")
c2.metric("Nombre de dépenses", len(dm))
c3.metric("Mois d'historique",
          df["date"].dt.to_period("M").nunique() if not df.empty else 0)
c4.metric("Abonnements actifs", len(actifs_abo))

st.markdown("---")

# ── INVENTAIRE ────────────────────────────────────────────────────────
st.subheader("▸ INVENTAIRE")

cols = st.columns(4)
for i, cat in enumerate(CATEGORIES):
    with cols[i%4]:
        x = dm[dm["categorie"]==cat]["montant"].sum()
        z, auto = budget_effectif(cat, bud, df)
        ratio   = (x/z) if z and z>0 else 0
        pers    = cat in CATEGORIES_PERSISTANTES

        if   ratio >= 1.0: couleur,classe = ROUGE,"alerte"
        elif ratio >= 0.8: couleur,classe = OR,   "warning"
        else:              couleur,classe = CYAN,  "persist" if pers else "ok"

        z_str  = f"{z:.0f}" if z else "∞"
        pct    = min(int(ratio*100),100)
        badge  = '<span class="badge-auto">AUTO</span>' if pers else ""
        auto_s = '<span style="font-size:.6em;color:#666688"> (moy)</span>' if auto and z else ""

        st.markdown(f"""
<div class="inv-card {classe}">
  <div class="inv-icone">{ICONES[cat]}</div>
  <div class="inv-nom">{cat} {badge}</div>
  <div class="inv-xz">{x:.0f} € / {z_str} €{auto_s}</div>
  <div class="inv-bar-bg">
    <div class="inv-bar" style="width:{pct}%;background:{couleur};"></div>
  </div>
</div>""", unsafe_allow_html=True)

        if st.button(f"⚙ {cat}", key=f"b_{cat}", use_container_width=True):
            st.session_state.popup = cat if st.session_state.popup != cat else None
            st.rerun()

st.markdown("---")

# ── PANNEAU CATÉGORIE ─────────────────────────────────────────────────
if st.session_state.popup:
    cat = st.session_state.popup
    pers = cat in CATEGORIES_PERSISTANTES

    st.subheader(f"{ICONES[cat]}  {cat.upper()}")

    labels = ["➕ Ajouter","🗑 Dépenses","💰 Budget"]
    if pers: labels.append("📡 Abonnements")
    tabs = st.tabs(labels)

    # ── Ajouter ──────────────────────────────────────────────────────
    with tabs[0]:
        with st.form(f"f_add_{cat}", clear_on_submit=True):
            ca, cb = st.columns(2)
            with ca:
                d_inp = st.date_input("Date", value=date.today())
                m_inp = st.number_input("Montant (€)", min_value=0.01, step=0.5, value=10.0)
            with cb:
                n_inp = st.text_input("Note", placeholder="ex : courses Lidl")
                p_inp = st.file_uploader("Photo / reçu",
                    type=["png","jpg","jpeg","webp"], key=f"pu_{cat}")
            ok = st.form_submit_button("⏎ Enregistrer")
        if ok:
            ph = sauvegarder_photo(p_inp, d_inp, cat) if p_inp else ""
            ajouter_depense(d_inp, cat, m_inp, n_inp, ph)
            st.success(f"✓  {m_inp:.2f} € ajouté en {cat}")
            st.rerun()

    # ── Gérer / supprimer dépenses ────────────────────────────────────
    with tabs[1]:
        dep_cat = df[df["categorie"]==cat].sort_values("date", ascending=False)
        if dep_cat.empty:
            st.info("Aucune dépense pour cette catégorie.")
        else:
            for idx, row in dep_cat.iterrows():
                ca,cb,cc,cd = st.columns([2,3,2,1])
                with ca: st.code(row["date"].strftime("%d/%m/%Y"), language=None)
                with cb:
                    note = f" · *{row['note']}*" if row.get("note") else ""
                    st.markdown(f"**{row['montant']:.2f} €**{note}")
                with cc:
                    if row.get("photo") and os.path.exists(str(row["photo"])):
                        ext = Path(str(row["photo"])).suffix.lower()
                        if ext in [".png",".jpg",".jpeg",".webp"]:
                            mime = "image/jpeg" if ext in [".jpg",".jpeg"] else f"image/{ext[1:]}"
                            b64  = img_b64(str(row["photo"]))
                            st.markdown(
                                f'<img src="data:{mime};base64,{b64}" class="photo-thumb">',
                                unsafe_allow_html=True)
                with cd:
                    if st.button("🗑", key=f"sd_{idx}", help="Supprimer"):
                        supprimer_depense(idx)
                        st.success("Supprimé.")
                        st.rerun()
                st.divider()

    # ── Budget Z ──────────────────────────────────────────────────────
    with tabs[2]:
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
            sauvegarder_budget(cat, nv if nv > 0 else None)
            st.success(f"Budget {cat} → {nv:.0f} €/mois")
            st.rerun()

    # ── Abonnements ───────────────────────────────────────────────────
    if pers and len(tabs) > 3:
        with tabs[3]:
            st.caption("Abonnements **actifs** = injectés automatiquement le 1er du mois.")
            abo = charger_abonnements()
            if abo:
                for nom, val in list(abo.items()):
                    actif = val.get("actif", True)
                    ca,cb,cc,cd = st.columns([3,2,1,1])
                    with ca: st.markdown(f"**{nom}**")
                    with cb: st.markdown(f"{val['montant']:.2f} €/mois")
                    with cc:
                        lbl = "⏸" if actif else "▶"
                        if st.button(lbl, key=f"tog_{nom}",
                                     help="Désactiver/Réactiver"):
                            abo[nom]["actif"] = not actif
                            sauvegarder_abonnements(abo)
                            st.rerun()
                    with cd:
                        if st.button("🗑", key=f"del_abo_{nom}",
                                     help="Supprimer définitivement"):
                            del abo[nom]
                            sauvegarder_abonnements(abo)
                            st.rerun()
                st.divider()
            else:
                st.info("Aucun abonnement enregistré.")

            st.markdown("**Ajouter un abonnement**")
            with st.form("f_abo_add", clear_on_submit=True):
                na, nb = st.columns(2)
                with na: nom_n = st.text_input("Nom", placeholder="Netflix…")
                with nb: mnt_n = st.number_input("€/mois", min_value=0.01,
                                                   step=0.5, value=10.0)
                ok3 = st.form_submit_button("⏎ Ajouter")
            if ok3 and nom_n.strip():
                abo[nom_n.strip()] = {"montant":float(mnt_n),"actif":True}
                sauvegarder_abonnements(abo)
                st.success(f"« {nom_n} » ajouté — {mnt_n:.2f} €/mois")
                st.rerun()

    if st.button("✕ Fermer", key="close_popup"):
        st.session_state.popup = None
        st.rerun()
    st.markdown("---")

# ── ALERTES ───────────────────────────────────────────────────────────
st.subheader("⚠ ALERTES")
alert = False
for cat in CATEGORIES:
    x = dm[dm["categorie"]==cat]["montant"].sum()
    if x == 0: continue
    z, auto = budget_effectif(cat, bud, df)
    if not z: continue
    r   = x/z if z > 0 else 0
    src = "moy. auto" if auto else "budget fixé"
    if r >= 1.0:
        st.error(f"🔴 **{cat}** — {x:.2f} € / {z:.2f} € ({src}) — dépassement +{(r-1)*100:.0f}%")
        alert = True
    elif r >= 0.8:
        st.warning(f"🟡 **{cat}** — {x:.2f} € / {z:.2f} € ({src}) — {r*100:.0f}% du seuil")
        alert = True
if not alert:
    st.success("🟢 Aucun seuil franchi ce mois-ci.")

st.markdown("---")

# ── GRAPHIQUES ────────────────────────────────────────────────────────
if not df.empty:
    plt.rcParams.update({
        "figure.facecolor":"#0E0E1A","axes.facecolor":"#0E0E1A",
        "axes.edgecolor":"#333355","axes.labelcolor":"#AAAACC",
        "text.color":"#CCCCEE","xtick.color":"#666688","ytick.color":"#666688",
        "grid.color":"#1E1E30","grid.linestyle":"--","grid.linewidth":0.5,
    })
    cg, cd = st.columns(2)
    with cg:
        st.subheader("▸ ÉVOLUTION MENSUELLE")
        df2 = df.copy()
        df2["mois"] = df2["date"].dt.to_period("M").astype(str)
        tot = df2.groupby("mois")["montant"].sum()
        fig,ax = plt.subplots(figsize=(6,3.5))
        tot.plot(kind="bar", ax=ax, color=VIOLET, width=0.65)
        ax.set_ylabel("€"); ax.set_xlabel("")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    with cd:
        st.subheader("▸ RÉPARTITION")
        tot_c = df.groupby("categorie")["montant"].sum().sort_values(ascending=False)
        clrs  = [VIOLET,CYAN,VERT,"#FF8C42",ROUGE,"#8888FF",OR,"#666688"]
        fig,ax = plt.subplots(figsize=(6,3.5))
        wedges,texts,autotexts = ax.pie(
            tot_c.values, labels=tot_c.index, autopct="%1.0f%%",
            colors=clrs[:len(tot_c)], textprops={"color":"#DDDDEE","fontsize":8},
            pctdistance=0.82, startangle=90)
        for at in autotexts: at.set_fontsize(7)
        plt.tight_layout()
        st.pyplot(fig)
    st.markdown("---")

# ── PROJECTION ML ─────────────────────────────────────────────────────
st.subheader("▸ PROJECTION // MOIS SUIVANT")
rows = []
for cat in CATEGORIES:
    if df.empty or df[df["categorie"]==cat].empty: continue
    p, meth = predire(df, cat)
    z, _    = budget_effectif(cat, bud, df)
    rows.append({
        "Catégorie": cat,
        "Projection (€)": f"{p:.2f}",
        "Méthode": meth,
        "Ref. budget (€)": f"{z:.2f}" if z else "—",
    })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("Pas assez de données.")
st.markdown("---")

# ── AGENT IA ──────────────────────────────────────────────────────────
st.subheader("🤖 AGENT")
q = st.text_input("Question en langage naturel",
    placeholder="Ex : Combien en Alimentation ce mois ?",
    key="q_agent")
if q:
    st.markdown(f"> {agent_repondre(q, df)}")
with st.expander("Exemples"):
    st.markdown("""
- « Combien j'ai dépensé en Alimentation ? »
- « Ma catégorie la plus chère ? »
- « Moyenne Transport »
- « Combien en Loisirs en mars ? »
""")
st.markdown("---")

# ── HISTORIQUE ────────────────────────────────────────────────────────
st.subheader("▸ HISTORIQUE")
if not df.empty:
    aff = df[["date","categorie","montant","note"]].sort_values("date",ascending=False).copy()
    aff["date"] = aff["date"].dt.strftime("%d/%m/%Y")
    st.dataframe(aff, use_container_width=True, hide_index=True)
else:
    st.info("Aucune dépense pour l'instant.")
st.markdown("---")

# ── IMPORT CSV ────────────────────────────────────────────────────────
st.subheader("▸ IMPORT CSV")
st.caption("Colonnes : date, categorie, montant (note et photo optionnels)")
up = st.file_uploader("Fichier CSV", type="csv")
if up:
    di = pd.read_csv(up, dtype=str)
    di["montant"] = pd.to_numeric(di["montant"], errors="coerce")
    di["date"]    = pd.to_datetime(di["date"],    errors="coerce")
    di = di.dropna(subset=["date","montant"])
    sauvegarder_depenses(
        pd.concat([charger_depenses(), di], ignore_index=True).drop_duplicates())
    st.success(f"✓  {len(di)} entrées importées.")
    st.rerun()
