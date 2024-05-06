
# Langgraph Courses 

1) What is langgraph ? 

2) What is langgraph used for ? 

3) How langgraph works ? 

4) Why langgraph ? 

5) In what can we use langgraph ? 

6) how to setup langgraph ? 



# Responses To all these Questions 


1) Rep N°1 : 
Langgraph is library for building stateful , multi-actor applications with LLMs .
The main used is to add cycles to your LLM application and that's allow to control the behavior .

LangGraph est une bibliothèque permettant de construire des applications multi-acteurs avec des LLMs, 
facilitant la coordination cyclique de multiples chaînes inspirées de Pregel et Apache Beam, 
avec une interface similaire à NetworkX, principalement utilisée pour ajouter des cycles aux applications 
LLM pour des comportements de type agent.


2) Rep N°2 : 

LangGraph utilise le concept central de l'état, où chaque exécution de graphe passe un état entre les nœuds,
 permettant aux nœuds de mettre à jour cet état avec leur valeur de retour.
  L'état peut être général, mais dans un exemple simplifié, 
  il est limité à une liste de messages de discussion, ce qui est pratique pour les modèles de discussion LangChain.

3) Rep N°3 :

LangGraph concept is based on diagram flow to control the behavior of the LLMs and chains can not do that .
 and the the central concept of langGraph is state each graph execution is passed between nodes
in the graph as they execute and each node updates this internal state with 
 its return value after it executes .
  The way that the graph updates its internal state is defined by either the type of graph or a custom function . 

4) Rep N°4 :

LangGraph because with chains we can not intervene in the execution process 
of the LLMs and that's why we have LangGraph .


5) Rep N°5 :

We can use langGraph in NLP applications , with crewai  with LLMs etc .It's 
also incorporated in the langchain packages . 

6) Rep N°6 : 

To set up langGraph you just need to run the following command : 

---------- Set up langGraph --------------- : 
   pip install langgraph 




**Qu'est-ce que LangGraph ?**

LangGraph est une librairie permettant de construire des applications multi-acteurs à état avec des modèles de langage volumineux (LLM).
 Il vous permet de coordonner plusieurs composants d'IA (acteurs) travaillant 
 ensemble sur plusieurs étapes de manière cyclique. Il s'inspire de Pregel et d'Apache Beam 
 et utilise une syntaxe de type NetworkX pour définir votre application.

**Quand utiliser LangGraph ?**

* Votre application LLM nécessite des cycles. LangChain ne fonctionnera pas car il ne peut gérer que des workflows linéaires (DAG).
* Votre application implique un comportement d'agent, où un LLM est appelé en boucle pour prendre des mesures en fonction de ses réponses.

**Concepts clés**

* **État:** Chaque exécution de graphe maintient un état qui est transmis entre les nœuds lors de leur exécution.
 Les nœuds peuvent mettre à jour cet état avec leurs valeurs de retour.

* **Nœuds:** Ce peuvent être des fonctions ou des éléments exécutables qui traitent l'état actuel et renvoient des mises à jour.
* **Arêtes:** Les arêtes connectent des nœuds, indiquant le flux de données à travers le graphe.
* **Arêtes conditionnelles:** L'exécution peut être acheminée en fonction de la sortie d'un nœud à l'aide d'instructions conditionnelles.

**Exemple : Agent utilisant LangChain et appel d'outils**

Cet exemple montre comment créer un agent qui utilise un LLM et des outils externes pour répondre aux messages des utilisateurs. 
L'agent maintient son état sous forme de liste de messages.


* **Outils:** Des services externes comme Tavily sont utilisés pour effectuer des actions en fonction des requêtes de l'agent.
* **LLM (ChatOpenAI):** C'est le modèle d'IA central utilisé pour la conversation et l'appel de fonctions.
* **StateGraph:** Cette classe gère l'état et l'exécution du graphe.

**Comment ça fonctionne ?**

1. L'utilisateur fournit un message initial.
2. Le nœud `call_model` est appelé, interrogeant le LLM pour une réponse ou un appel de fonction.
3. Si la réponse du LLM implique un appel de fonction :
   - Le nœud `call_tool` est appelé, en utilisant l'outil spécifié dans l'appel de fonction.
   - La réponse de l'outil est intégrée à l'état de l'agent.
4. S'il n'y a plus d'appels de fonction :
   - La réponse du LLM est utilisée pour répondre à l'utilisateur.

**Avantages de LangGraph**

* **Streaming:** Obtenez des résultats tels qu'ils sont produits par chaque nœud du graphe.
* **Jetons LLM en streaming:** Accédez aux sorties LLM au fur et à mesure de leur génération.
* **Persistance:** Enregistrez l'état du graphe et reprenez l'exécution plus tard.
* **Humain dans la boucle:** Intégrez la révision humaine dans le workflow.
* **Visualisation:** Générez des représentations visuelles du graphe pour une meilleure compréhension.
* **Débogage "Voyage dans le temps" :** Accédez à n'importe quel point de l'exécution du graphe, modifiez l'état et reprenez l'exécution à partir de là.

**Information additionnelle**

Le document fournit également des références à divers cas d'utilisation et fonctionnalités de LangGraph, notamment :

* **Exécuteurs d'agent:** Composants prédéfinis pour diverses architectures d'agent.
* **Exemples d'agents de planification:** Agents qui planifient et exécutent des tâches à l'aide de LLM et d'outils externes.
* **Exemples multi-agents:** Comment créer plusieurs agents en collaboration.
* **Recherche Web:** Exemples d'utilisation de LangGraph pour la navigation Web et le traitement de données.
* **Documentation:** Explications détaillées des composants et fonctionnalités de LangGraph.
