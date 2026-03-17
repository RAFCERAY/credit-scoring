# Scoring de Crédit Bancaire — Modélisation Prédictive

## Objectif
Construire un modèle prédictif pour classifier les clients bancaires en **Bon client** ou **Mauvais client** afin d'automatiser les décisions d'octroi de crédit.

## Dataset
- 468 clients bancaires
- 15 variables (comportementales, financières, démographiques)
- Dataset parfaitement équilibré : 50.6% bons clients / 49.4% mauvais clients

## Méthodologie
1. Exploration et visualisation des données
2. Préparation — encodage, train/test split (80/20), standardisation
3. Modélisation — Régression Logistique vs Random Forest
4. Évaluation — AUC-ROC, matrice de confusion, validation croisée
5. Interprétation des variables et recommandations métier

## Résultats

| Modèle | Accuracy | AUC-ROC | F1 moyen |
|--------|----------|---------|----------|
| Régression Logistique | **79.8%** | 0.841 | **0.80** |
| Random Forest | 78.7% | **0.857** | 0.79 |

**Modèle retenu : Régression Logistique** — meilleures performances globales et interprétabilité essentielle pour la conformité réglementaire bancaire.

## Variables les plus prédictives
- Score 1 et Score 2 (scores internes)
- Domiciliation du salaire
- Historique de chéquier
- Ancienneté client

## Structure du projet
```
credit-scoring/
├── data/               ← dataset
├── notebooks/          ← notebook Jupyter complet
├── outputs/
│   ├── figures/        ← 7 graphiques d'analyse
│   └── models/         ← modèles sauvegardés (.pkl)
└── README.md
```

## Stack technique
Python · pandas · scikit-learn · statsmodels · matplotlib · seaborn

## Auteur
Rafika Ayari Cervera — [LinkedIn](https://www.linkedin.com/in/rafikacervera/)
