<a id="readme-top"></a>

# Projet 7
Concevez et déployez un système RAG

## A propos du projet 
Ce projet consiste à concevoir, développer et évaluer un Proof of Concept (POC) complet qui doit permettre aux utilisateurs de poser des questions en langage naturel sur la programmation culturelle et d'obtenir des réponses précises, factuelles et actualisées, basées sur les données ouvertes de la plateforme Open Agenda.

Le développement a été effectué sur un système d'exploitation Windows.

Le projet comprend :
- un pipeline d'injection et de vectorisation des événements culturels
- un pipeline RAG 
- une API permettant d'intéragir avec le RAG
- une interface client (streamlit) pour tester l'application
- les tests unitaires
- l'évaluation du RAG
- un pipeline CI
- des fichiers Dockerfile et docker-compose pour la conteneurisation de l'application

<img src="./docs/images/Diagramme UML architecture.png" alt="Architecture projet" width="650">


> Pour plus de détails, veuillez vous référer à au [rapport technique](docs/Rapport technique.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Programmation

   * [Python V3.12][python-url]
   * [Package manager : uv][uv-url]
   * [Versionning : GIT][git-url]

### Plateformes

   * [VS Code][vscode-url]
   * [Docker desktop][docker-url]
   * [Github][github-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prérequis

   Avant de commencer le projet, veuillez préparer correctement votre environnement de développement.

      1. Installez Python version V3.12
      2. Installez Visual Studio Code, ainsi que ses extensions Python (python, python debugger, ...)
      3. Installez Docker desktop
      4. Installez le système de contrôle de version Git
      5. Installez le manager de packages et d'environnements UV

### Installation package manager UV

Ouvrez la console Powershell et tapez la commande suivante pour installer UV :

   ```sh
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
### Cloner le projet depuis le repo GitHub

Pour commencer à travailler en local sur le projet, veuillez cloner celui-ci avec la commande suivante (via GIT bash) :

   ```sh
   git clone https://github.com/clementghoris-cpu/Projet-7.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>   

## Organisation des branches
- La documentation se fait dans les branches `docs`
- La branche principale de développement est `develop`. Les nouvelles fonctionnalités sont développées directement dans celle-ci
- la branche `main` sert au déploiement sur l'environnement de *production*. Elle est mise à jour automatiquement après une pull request de la branche develop vers main et que les tests unitaires soient validés

### Convention pour les commits
La structure d'un message de commit se fait de la façon suivante :

```sh
   <type> (cible) : description

   Exemple : feat (api) : modifier le format de la reponse de l'endpoint /predict
```

#### Les principaux types
- **`feat`** : Nouvelle fonctionnalité
- **`fix`** : Correction d'un bug
- **`docs`** : Changement dans la documentation
- **`refactor`** : Modification du code qui ne corrige rien et n'ajoute rien
- **`test`** : Ajout ou modification de tests
- **`chore`** : Tâches répétitives, mise à jour de dépendances, configuration des outils

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Déploiement de l'application

Le déploiement de l'application se fait dans les conteneurs Docker: Nous avons au total 2 conteneurs qui peuvent être utilisés :
- **API** : qui est l'application permettant d'exposer le pipeline RAG
- **streamlit** : qui est l'interface client (chabot) permettant de tester l'application

Chaque image est configuré via un Dockerfile :
- se situant dans *src/api/* pour le conteneur API
- se situant dans *src/client/* pour le conteneur Client

La configuration des services se fait dans le fichier docker-compose.yaml.

Pour déployer les containers, utilisez la commande :
   ```sh
   docker compose -f docker-compose.yml up -d [api] [streamlit] --build
   ```

Pour démarrer la totalité des conteneurs, il n'est pas utile de spécifier chaque nom de container. `--build` permet de recompiler les conteneurs après une modification de code.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Variables d'environnement
Les variables d'environnements sont définies dans un fichier .env qui n'est pas intégré dans le dépôt Github pour éviter les fuites de données (Clés API). Il y a néanmoins un fichier ***.env.example***.

```sh
    # Utiliser la commande suivante pour créer un fichier .env à partir du fichier exemple
    cp .env.example .env
```

| Variable | Description | Utilisation |
| --- | --- | --- |
| **API_KEY_MISTRAL** | Clé d'accès à l'API Mistral | `src/rag/rag_chain.py`, `tests/evaluation/rag_evaluation.py` |
| **RAG_API_URL** | Adresse URL de l'API | `src/client/chatbot.py` |
| **RAG_API_PORT** | Port de l'API | `src/client/chatbot.py` |
| **OPENAGENDA_API_URL** | Adresse URL de l'API OpenAgenda | `src/utils/data_fetch.py` |
| **OPENAGENDA_LOCATION_FILTER** | Filtre appliqué pour récupérer les événements | `src/utils/data_fetch.py` |
| **OPENAGENDA_START_DATE** | Date à partir de laquelle récupérer les événements | `src/utils/data_fetch.py` |
| **OPENAGENDA_END_DATE** | Date jusqu'à laquelle récupérer les événements | `src/utils/data_fetch.py` |
| **OPENAGENDA_MAX_EVENTS** | Maximum d'événement à récupérer | `src/utils/data_fetch.py` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tests unitaires et évaluation RAG

### Workflow CI (continuous integration)

La vérification de la qualité du code avec [Ruff][ruff-url], les tests untaires et l'évaluation du RAG sont exécutés automatiquement via le workflow CI (*.github/workflow/ci.yaml*).

### Tests unitaires

Les tests unitaires peuvent être également exécutés de façon manuel via Powershell :

```sh
    uv run pytest .\tests\units\                                  # Lancer l'ensemble des tests unitaires
    uv run pytest .\tests\units\<dossier>\<nom fichier>.py        # Lancer un seul fichier de tests
```

### Evaluation RAG

Pour lancer manuellement l'évaluation du RAG, veuillez taper la commande suivante dans Powershell :

```sh
    uv run python -m tests.evaluation.rag_evaluation
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

[python-url]:https://www.python.org/
[uv-url]:https://docs.astral.sh/uv/
[git-url]:https://git-scm.com/
[vscode-url]:https://code.visualstudio.com/download?_exp_download=fb315fc982
[docker-url]:https://www.docker.com/products/docker-desktop/
[github-url]:https://github.com/
[streamlit-url]:https://streamlit.io/
[pytest-url]:https://docs.pytest.org/en/stable/
[ruff-url]:https://docs.astral.sh/ruff/linter/