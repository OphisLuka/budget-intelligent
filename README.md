# 💶 Budget personnel intelligent

Application web (Streamlit) de suivi de dépenses personnelles, avec alertes de dépassement de budget, **prédiction des dépenses du mois prochain par catégorie** (régression linéaire, scikit-learn), et un **agent d'analyse en langage naturel**.

## Fonctionnalités

- **Saisie de dépenses** via un formulaire (date, catégorie, montant), avec persistance locale
- **Budgets par catégorie**, au choix :
  - fixés manuellement par l'utilisateur, ou
  - calculés automatiquement comme la moyenne mensuelle de l'historique
- **Alertes de dépassement** sur le mois en cours
- **Prédiction du mois prochain par catégorie** : régression linéaire si l'historique est suffisant (≥ 4 mois), repli automatique sur une simple moyenne sinon
- **Agent d'analyse en langage naturel** (voir ci-dessous)
- **Visualisations** : évolution mensuelle des dépenses, répartition par catégorie
- **Import CSV** pour ajouter un historique existant en une fois
- Toutes les données restent **locales**

## 🤖 L'agent d'analyse

Le fichier `agent.py` permet de poser des questions en langage naturel sur ses propres dépenses :

- *« Combien j'ai dépensé en Alimentation ? »*
- *« Quelle est ma catégorie la plus chère ? »*
- *« Moyenne Loisirs »*
- *« J'ai dépensé 20€ en Transport »* (aide à structurer une saisie, sans jamais l'enregistrer automatiquement)

### Ce que cet agent EST et N'EST PAS

**Ce n'est pas un grand modèle de langage (LLM)** comme GPT ou Claude. C'est un agent **« rule-based »** : il reconnaît des mots-clés et des motifs (regex) dans la question posée, puis va chercher la réponse exacte dans les données réellement enregistrées. Il ne génère jamais de texte « inventé » — soit il trouve une réponse précise dans les données, soit il dit explicitement qu'il n'a pas compris.

**Pourquoi ce choix plutôt qu'un vrai LLM ?**
- Aucune clé API à configurer : n'importe qui peut cloner le projet et le faire fonctionner immédiatement
- Aucun coût, aucune dépendance à un service externe
- 100% local : les données financières ne quittent jamais la machine

**Garde-fou volontaire** : l'agent peut aider à *comprendre* une intention de dépense exprimée en langage naturel, mais il n'enregistre **jamais** de donnée de lui-même. Toute écriture dans les fichiers de données passe obligatoirement par une action explicite de l'utilisateur (le formulaire). C'est un choix de conception délibéré : un agent qui analyse ne doit pas être celui qui décide d'agir.

L'architecture (une fonction `repondre(question, donnees)` isolée dans son propre fichier) permettrait de remplacer ce moteur par un vrai appel à une API de LLM dans une future version, sans toucher au reste de l'application.

## Aperçu

*(à ajouter : captures d'écran de l'application)*

## Installation

```bash
git clone https://github.com/OphisLuka/budget-intelligent.git
cd budget-intelligent
pip install -r requirements.txt
```

## Utilisation

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

## Méthode de prédiction

Pour chaque catégorie, les dépenses sont agrégées par mois. Si au moins 4 mois de données existent, un modèle de régression linéaire est entraîné (le numéro du mois comme variable explicative, le total dépensé comme cible), et utilisé pour estimer le mois suivant. En dessous de ce seuil, le modèle de régression serait peu fiable (trop peu de points), donc l'application utilise une simple moyenne historique à la place.

## Stack technique

- Python
- [Streamlit](https://streamlit.io/) — interface web interactive
- [pandas](https://pandas.pydata.org/) — manipulation des données
- [scikit-learn](https://scikit-learn.org/) — régression linéaire pour la prédiction
- [Matplotlib](https://matplotlib.org/) — visualisations
- Agent rule-based maison (`agent.py`) — aucune dépendance externe

## Limites et pistes d'amélioration

- La régression linéaire suppose une tendance simple — elle ne capte pas de saisonnalité
- L'agent rule-based ne comprend qu'un nombre limité de formulations — il n'a pas la flexibilité d'un vrai LLM
- Pistes futures : modèles de séries temporelles plus avancés, brancher une vraie API de LLM en option (avec clé API fournie par l'utilisateur), déploiement en ligne (Streamlit Community Cloud)

## Auteur

[OphisLuka]

