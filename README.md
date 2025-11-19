# 🌦️ Jena Climate — Prévision Temporelle par Deep Learning

Modélisation et prédiction de la température à partir du **dataset climatique Jena Climate** grâce à différents modèles de deep learning : Dense, CNN, RNN, LSTM, GRU, et comparaison avec un modèle naïf.


## 📌 **Objectif du Projet**

L’objectif est de **prédire la température future ("T (degC)")** à partir de séries temporelles multivariées enregistrées toutes les 10 minutes entre 2009 et 2016.

Le projet inclut :

* Nettoyage et analyse exploratoire du dataset
* Sélection de variables via matrice de corrélations
* Normalisation et préparation séquentielle des données
* Création de jeux d'entraînement / validation / test
* Entraînement de multiples architectures deep learning :

  * Régression linéaire (1 neurone)
  * Réseau dense profond
  * CNN 1D
  * RNN
  * LSTM
  * GRU
* Comparaison des modèles via MAE
* Benchmark avec modèle naïf


## 📂 **Dataset : Jena Climate**

📄 Source : *jena_climate_2009_2016.csv*

Nombre de variables originales : **15**
Exemples de features :

| Variable  | Description         |
| --------- | ------------------- |
| T (degC)  | Température         |
| p (mbar)  | Pression            |
| rh (%)    | Humidité            |
| wd (deg)  | Direction du vent   |
| wv (m/s)  | Vitesse du vent     |
| Radiation | Rayonnement solaire |


## 🧹 **Prétraitement & Sélection des Variables**

### ✔️ Conversion et Indexation

* Transformation de `Date Time` → format datetime
* Mise en index temporel

### ✔️ Analyse de corrélation

* Construction de la matrice
* Suppression automatique des variables fortement corrélées (>|0.85|)

Variables supprimées :

```
["Tpot (K)", "Tdew (degC)", "VPact (mbar)", "VPmax (mbar)", "max. wv (m/s)"]
```

### ✔️ Normalisation

Standardisation min-max sur chaque split (train/val/test).


## 🧪 **Construction des Jeux de Données**

Le modèle utilise une **fenêtre temporelle glissante** :

* **5 jours d’historique**
* **1 jour futur à prédire**
* Fréquence retenue : 1 mesure / heure (step=6)

Pipeline TensorFlow :

```python
timeseries_dataset_from_array(...)
```

Splits :

* **Train : 60%**
* **Validation : 20%**
* **Test : 20%**


## 🤖 **Modèles Testés**

Chaque modèle prédit la température future à partir des données passées.

### 🔹 1. Modèle naïf (baseline)

Décale simplement la série d’une journée.
Permet d’évaluer si les modèles sont réellement utiles.

### 🔹 2. Régression linéaire à 1 neurone

```python
Dense(1, activation="linear")
```

### 🔹 3. Réseau dense profond (Fully Connected)

Plusieurs couches denses + ReLU
→ Performances limitées car structure peu adaptée aux séries temporelles.

### 🔹 4. CNN 1D

* Extraction de motifs locaux temporels
* Convolutions + max pooling
* Architecture légère mais performante

### 🔹 5. RNN classique

* 2 couches SimpleRNN (return_sequences = True/False)

### 🔹 6. LSTM

* Capable de gérer dépendances longues
* Plus lourd à entraîner

### 🔹 7. GRU

* Alternative plus légère au LSTM
* Souvent meilleur compromis


## 📊 **Résultats & Comparaison**

| Modèle               | MAE ↓    |
| -------------------- | -------- |
| **Modèle naïf**      | 2.59     |
| Régression 1 neurone | 3.82     |
| Dense profond        | 5.41     |
| CNN                  | 3.35     |
| RNN                  | 2.25     |
| LSTM                 | 10.32    |
| **GRU (meilleur)**   | **1.68** |

➡️ **Le GRU surpasse tous les autres modèles, y compris le modèle naïf, avec la meilleure MAE.**


## 🛠️ **Technologies & Librairies**

* Python 3.x
* NumPy
* Pandas
* Matplotlib / Seaborn
* TensorFlow / Keras
* Scikit-learn


## ▶️ **Exécution du Projet**

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/username/jena-climate-deep-learning.git
cd jena-climate-deep-learning
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer l'entraînement

Notebook :

```
Jena_Climate_DeepLearning.ipynb
```

Ou script :

```bash
python train_models.py
```


## 📈 **Visualisations incluses**

* Heatmap des corrélations
* Courbes d’apprentissage (loss & MAE)
* Comparaison prédictions / valeurs réelles
* Analyse des modèles


## 🔮 **Améliorations Futures**

* Ajout d’un modèle **Transformer pour séries temporelles**
* Optimisation automatique (KerasTuner)
* Prévision multi-pas (multi-step forecasting)
* Modèles hybrides : CNN + LSTM
* Déploiement via FastAPI ou Streamlit


## 👤 **Auteur**

**Alex Alkhatib**
Projet Deep Learning — Prévision Temporelle


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
