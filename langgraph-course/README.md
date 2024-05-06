#### 1. What is Langgraph ?

Langgraph est une bibliotheque permettant de créer desa applications mult-acteurs avec état avec des LLM , construites sur Langchain . 
Son principal objectif est d'ajouter des cycles à votre application LLM . 

Les Cycles sont importants  pour les comportements de type agent , ou vous appelez un LLM en boucle , en lui demandant quelle action effectuer ensuite .

Mots clés : 
Application multi-acteurs ,
ajouter des cycles a votre application LLM , 
les cycles permettent de controller le flux d'execution de notre LLM .  

L'un des concepts centraux de LangGraph est l'état . Chaque exécution de graph cée un état qui transmis entre les noeuds du graph au fur et à mesure de leur execution, chaque noeud met à jour cet état interne avec sa valeur de retour après Execution . La facon dont le graph  met à jour son état interne est définie soit par le type de graph choisi , soit par une fonction personnalisée . 

# Qu'est qu'il faut retenir en resumé sur la comprehension de LangGraph ? 

Langgraph c'est une bibliothèque de langchain qui nous permet de controller l'execution notre application LLM , en initialisant un flux de travail qui sera composé de nodes contenant chacun un état et chaque node retourne a la sortie met a jour son état avec sa valeur de retour apres son execution . La façon dont le graph met a jour son état est soit definie soit par le type de graph soit par une fonction personnalisée . 

