# Defining Your Own Tool 

In this folder we gonna learn all about Langchain in order to build our own tool and how to  build our own Agents 


# What do we need to build a tool in Langchain 

Tool consists of several components 

**name**  : this is required and must be unique 

**description** : is optional but recommended , as it is used by an agent to determine tool use 



**args_schema** : (Pydantic BaseModel) -> permet de definir les structures des donnés et de valider les sorties des models de langage  (LLM)


 *Pydantic BaseModel*  : Fonctionnement 

1. Définir une structure de données Pydantic:
Commencez par créer une classe Pydantic BaseModel qui représente la structure des données que vous souhaitez obtenir du modèle de langage. Cette classe définit les champs, leurs types de données et les contraintes de validation.

2. Créer un analyseur Pydantic:

Ensuite, créez un objet PydanticOutputParser en spécifiant la classe Pydantic BaseModel comme paramètre. Cet analyseur sera utilisé pour convertir les sorties du modèle de langage en instances de la classe Pydantic BaseModel.

3. Construire une chaîne de traitement:

Utilisez le modèle de langage de votre choix et l'analyseur Pydantic pour construire une chaîne de traitement LangChain. Cette chaîne définira le flux d'exécution, en commençant par l'invite du modèle de langage et en terminant par l'analyse de la sortie.

4. Exécuter la chaîne de traitement:
Invoquez la chaîne de traitement en fournissant les données d'entrée nécessaires. L'analyseur Pydantic sera alors appliqué à la sortie du modèle de langage, garantissant qu'elle correspond à la structure de données Pydantic définie.

5. Model I/O: Facilitates interaction with various language models, handling their inputs and outputs efficiently.
Retrieval: Enables access to and interaction with application-specific data, crucial for dynamic data utilization.

6. Agents: Empower applications to select appropriate tools based on high-level directives, enhancing decision-making capabilities.
7. Chains: Offers pre-defined, reusable compositions that serve as building blocks for application development.
Memory: Maintains application state across multiple chain executions, essential for context-aware interactions.

8. **LangServe** steps in as a versatile library for deploying LangChain chains as REST APIs.

9. **LangSmith** serves as a developer platform. It's designed to debug, test, evaluate, and monitor chains built on any LLM framework. 

10. there are the LangChain Libraries, available in both Python and JavaScript. These libraries are the backbone of LangChain, offering interfaces and integrations for various components.

11. **Model I/O**: Facilitates interaction with various language models, handling their inputs and outputs efficiently.

12. **Retrieval**: Enables access to and interaction with application-specific data, crucial for dynamic data utilization.


stream: stream back chunks of the response
invoke: call the chain on an input
batch: call the chain on a list of inputs

stream : retransmettre en continu des morceaux de la réponse
invoke : appelle la chaîne sur une entrée
batch : appelle la chaîne sur une liste d'entrées

Dans LangChain, les termes stream (flux), batch (traitement par lots) et invoke (invoquer) correspondent à différentes manières d'interagir avec les modèles de langage volumineux (LLM) via les fonctionnalités de LangChain. Voici une explication de chaque concept et son importance pour les LLM :

1. Stream (Flux)

Concept: Le traitement en flux permet d'envoyer des données au LLM en continu, un élément à la fois.
Importance pour les LLM: Le traitement en flux est idéal pour les scénarios où vous avez une séquence de points de données ou une conversation en temps réel. Par exemple, imaginez la construction d'un chatbot qui doit répondre à chaque message utilisateur individuellement. En utilisant le traitement en flux, vous pouvez envoyer chaque message au LLM dès qu'il arrive, permettant une conversation plus naturelle et fluide.
Avantages:
Permet des interactions en temps réel ou quasi-temps réel avec le LLM.
Utile pour traiter de grands ensembles de données par morceaux, ce qui peut réduire l'utilisation de la mémoire par rapport au chargement de tout en une fois.
2. Batch (Traitement par lots)

Concept: Le traitement par lots consiste à envoyer un groupe de points de données au LLM en une seule fois. Il s'agit généralement d'un ensemble d'invites ou de questions liées entre elles.
Importance pour les LLM: Le traitement par lots est efficace pour gérer plusieurs tâches connexes simultanément. Par exemple, imaginez la génération de résumés d'une série d'articles de presse. Vous pouvez regrouper les articles et les envoyer par lots au LLM, ce qui peut améliorer l'efficacité par rapport à leur envoi un par un.
Avantages:
Peut être plus rapide que le traitement en flux pour un nombre fixe d'invites, surtout si le LLM gère efficacement le traitement par lots.
Peut être plus rentable dans certains cas car vous effectuez moins d'appels au LLM.
3. Invoke (Invoquer)

Concept: Le terme général pour envoyer une requête au LLM, qu'il s'agisse d'une seule invite, d'un flux de données ou d'un lot de points de données.
Importance pour les LLM: Invoke est la fonctionnalité principale qui vous permet d'interagir avec le LLM et d'obtenir sa réponse. Il englobe à la fois les méthodes de traitement en flux et par lots.
Choisir la bonne méthode

La meilleure méthode (flux, lot ou une combinaison) dépend de votre cas d'utilisation spécifique :

Interactions en temps réel ou traitement de grands ensembles de données en séquence: Utilisez le traitement en flux.
Envoi de plusieurs invites ou questions liées pour plus d'efficacité: Utilisez le traitement par lots.
Interaction générique avec le LLM: Utilisez invoke comme terme générique pour les traitements en flux et par lots





**Retrieval** 

Retrieval plays a crucial role un applications that require user specific data, not included in the model's training set . 
Langchain provides comprehensive suite of tools and functionalities to facilitate this process , catering to both simple and complex applications . 


Here is a breakdown of the components that plays an important role in Retrieval process 

**Document Loaders**

Document loaders in LangChain enable the extraction of data from various sources. With over 100 loaders available, they support a range of document types, apps and sources (private s3 buckets, public websites, databases).






## Résumé amélioré de la compréhension de LangChain

**Qu'est-ce que LangChain ?**

LangChain est une plateforme puissante pour la construction d'applications basées sur des modèles de langage volumineux (LLM). Il offre un ensemble complet d'outils et de fonctionnalités pour faciliter le développement, le déploiement et la gestion d'applications alimentées par LLM.

**Composants clés de LangChain**

* **Agents**: Les agents sont des composants intelligents qui sélectionnent les outils LangChain appropriés en fonction d'instructions de haut niveau. Cela permet aux applications de prendre des décisions plus efficaces et d'offrir des expériences utilisateur plus personnalisées.

* **Chaînes**: Les chaînes sont des workflows pré-définis et réutilisables qui servent de modules de base pour le développement d'applications. Elles permettent de combiner différents outils LangChain de manière efficace et flexible.

* **Outils**: Les outils sont des composants modulaires qui effectuent des tâches spécifiques, telles que la génération de texte, la traduction, la classification et l'analyse du sentiment. Ils peuvent être combinés et personnalisés pour créer des applications puissantes.

* **Modèle I/O**: Le modèle I/O facilite l'interaction avec divers LLM, en gérant efficacement leurs entrées et sorties. Cela permet d'intégrer de manière transparente les LLM dans les applications LangChain.

* **Récupération**: La récupération permet d'accéder aux données spécifiques à l'application et d'interagir avec elles. Cela est crucial pour les applications qui nécessitent des données dynamiques pour fonctionner efficacement.

* **LangServe**: LangServe est une bibliothèque polyvalente pour le déploiement de chaînes LangChain sous forme d'API REST. Elle permet d'exposer les fonctionnalités de LangChain aux applications externes et aux services Web.

* **LangSmith**: LangSmith est une plateforme de développement pour LangChain. Elle offre des outils pour le débogage, le test, l'évaluation et la surveillance des chaînes construites sur n'importe quel framework LLM.

* **Bibliothèques LangChain**: Les bibliothèques LangChain sont disponibles en Python et en JavaScript. Elles offrent des interfaces et des intégrations pour divers composants LangChain, simplifiant le développement d'applications.

## Points clés à retenir

* LangChain est une plateforme puissante pour la construction d'applications basées sur LLM.
* Elle offre un ensemble complet d'outils et de fonctionnalités pour faciliter le développement, le déploiement et la gestion d'applications alimentées par LLM.

* Les composants clés de LangChain incluent les agents, les chaînes, les outils, le modèle I/O, la récupération, LangServe, LangSmith et les bibliothèques LangChain.

* LangChain permet de créer des applications intelligentes et personnalisées qui peuvent effectuer diverses tâches, telles que la génération de texte, la traduction, la classification et l'analyse du sentiment.



## Résumé clair du texte sur la création d'outils et d'agents dans LangChain

**Créer un outil dans LangChain**

Un outil LangChain est un composant modulaire qui effectue une tâche spécifique, comme la génération de texte, la traduction ou la classification. Pour créer un outil, vous devez définir les éléments suivants :

* **Nom**: Le nom de l'outil, qui doit être unique.
* **Description**: Une description facultative de l'outil.
* **Schéma d'arguments**: Une structure de données Pydantic qui définit les entrées et les sorties de l'outil.
* **Code**: Le code Python qui implémente la logique de l'outil.

**Fonctionnement du schéma d'arguments Pydantic**

1. Définir une classe Pydantic BaseModel qui représente la structure des données que vous souhaitez obtenir du modèle de langage.
2. Créer un objet PydanticOutputParser en spécifiant la classe Pydantic BaseModel comme paramètre.
3. Construire une chaîne de traitement LangChain en utilisant le modèle de langage de votre choix et l'analyseur Pydantic.
4. Exécuter la chaîne de traitement en fournissant les données d'entrée nécessaires.

**Autres composants clés de LangChain**

* **Agents**: Les agents sélectionnent les outils LangChain appropriés en fonction d'instructions de haut niveau.
* **Chaînes**: Les chaînes sont des workflows pré-définis et réutilisables qui servent de modules de base pour le développement d'applications.
* **Modèle I/O**: Le modèle I/O facilite l'interaction avec divers modèles de langage.
* **Récupération**: La récupération permet d'accéder aux données spécifiques à l'application et d'interagir avec elles.
* **LangServe**: LangServe permet de déployer des chaînes LangChain en tant qu'API REST.
* **LangSmith**: LangSmith est une plateforme de développement pour LangChain.
* **Bibliothèques LangChain**: Les bibliothèques LangChain offrent des interfaces et des intégrations pour divers composants.

**Modes d'interaction avec les modèles de langage**

* **Flux**: Envoyer des données au modèle de langage en continu, un élément à la fois.
* **Traitement par lots**: Envoyer un groupe de points de données au modèle de langage en une seule fois.
* **Invoquer**: Envoyer une requête au modèle de langage, qu'il s'agisse d'une seule invite, d'un flux de données ou d'un lot de points de données.

**Récupération de données**

LangChain fournit un ensemble complet d'outils et de fonctionnalités pour faciliter le processus de récupération de données, y compris les chargeurs de documents qui permettent d'extraire des données de diverses sources.

## Améliorations par rapport au résumé original

* Le résumé est plus concis et plus facile à lire.
* Les termes clés sont définis clairement et concisement.
* La structure du résumé est logique et cohérente.
* Des exemples concrets ont été ajoutés pour illustrer les concepts clés.
* Le langage a été simplifié pour le rendre plus accessible à un public plus large.

## J'espère que ce résumé clair vous sera utile. N'hésitez pas à me contacter si vous avez d'autres questions.
