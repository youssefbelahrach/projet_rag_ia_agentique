# Agentic RAG : Architecture Cloud & DevOps

Ce projet implémente un système **RAG Agentique (Retrieval-Augmented Generation)**. Contrairement à un RAG classique, cet agent utilise **LangGraph** pour raisonner sur la pertinence des documents récupérés en local et décide de manière autonome s'il doit interroger le Web pour compléter ses connaissances.

Le domaine d'application de cette implémentation est l'**Informatique (Architecture Cloud et DevOps)**.

## Fonctionnalités

- **Recherche Vectorielle Locale** : Indexation et recherche de documents internes via **ChromaDB**.
- **Évaluation Agentique (Grader)** : Un LLM évalue la pertinence des informations locales récupérées vis-à-vis de la question de l'utilisateur.
- **Fallback Web Automatique** : Si les documents locaux sont insuffisants, l'agent utilise **Tavily API** pour effectuer une recherche sur Internet.
- **Gestion d'État** : Contrôle précis du flux d'exécution et de la mémoire via l'architecture orientée graphe de **LangGraph**.
- **Environnement Rapide** : Gestion des dépendances ultra-rapide avec l'outil **`uv`**.

## Technologies Utilisées

- **LangGraph & LangChain** : Orchestration de l'agent et du graphe d'états.
- **OpenAI (GPT-4o-mini & Text-Embedding-3-small)** : Modèles de langage et d'embeddings.
- **ChromaDB** : Base de données vectorielle locale.
- **Tavily Search API** : Outil de recherche Web optimisé pour les LLMs.
- **uv** : Gestionnaire de paquets et d'environnements virtuels Python.

## Structure du Projet

```text
agentic_rag_project/
├── data/
│   └── cloud_devops.txt           # Base de connaissances locale
├── .env                       # Fichier des variables d'environnement (non versionné)
├── main.py                    # Définition du graphe d'états et du workflow LangGraph
├── evaluate.py                # Script de test automatisé (Questions simples et complexes)
├── graph_architecture.png     # Visualisation du graphe LangGraph (générée automatiquement)
└── README.md                  # Documentation du projet
```

## Configuration de l'environnement

**Créer l'environnement virtuel**

```bash
uv venv
```

**Activer l'environnement (macOS/Linux)**

```bash
source .venv/bin/activate
```

**Activer l'environnement (Windows)**

```bash
.venv\Scripts\activate`
```

**Installer les dépendances**

```bash
uv add langgraph langchain langchain-openai langchain-community langchain-chroma tavily-python python-dotenv
```

**Variables d'environnement**
Créez un fichier .env à la racine du projet et ajoutez vos clés API :

```bash
OPENAI_API_KEY=sk-votre-cle-openai-ici
TAVILY_API_KEY=tvly-votre-cle-tavily-ici
```

## Utilisation et Évaluation

Le projet est livré avec un script d'évaluation prêt à l'emploi qui teste l'agent sur 20 questions (10 questions simples et 10 questions complexes nécessitant une recherche web).

Pour lancer l'évaluation :

```bash
uv run evaluate.py
```
