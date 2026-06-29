import time
from main import app


questions_simples = [
    "Qu'est-ce que le modèle IaaS et quelles responsabilités garde généralement l'équipe cliente ?",
    "Quelle est la différence entre une image et un conteneur Docker ?",
    "Quel est le rôle d'un Dockerfile ?",
    "À quoi sert le fichier .tfstate dans Terraform ?",
    "Quelle différence existe entre CI et CD ?",
    "Qu'est-ce que S3 Standard-IA ?",
    "Que signifie le principe de moindre privilège ?",
    "Quel est l'avantage du déploiement Blue-Green ?",
    "Combien d'environnements tournent en Blue-Green ?",
    "Pour quel type de données utiliser S3 Standard-IA ?",
]

questions_complexes = [
    "Comment mettre en place un pipeline CI/CD sécurisé sur Gitlab avec Terraform et Vault ?",
    "Expliquez l'architecture d'un cluster Kubernetes multi-région avec Cilium.",
    "Quels sont les coûts cachés lors de l'utilisation de clusters EKS sur AWS ?",
    "Quels sont les avantages de LangGraph par rapport à LangChain create_agent pour les workflows ?",
    "Comparez les performances entre AWS Lambda et Google Cloud Run en 2025.",
    "Comment gérer le state Terraform dans une équipe de 50 développeurs ?",
    "Conçois une pipeline CI/CD sécurisée pour une API conteneurisée déployée sur Kubernetes.",
    "Propose un plan de reprise pour une base de données critique avec une contrainte de perte de données limitée.",
    "Quelle est la meilleure stratégie pour migrer une base de données monolithique vers des microservices sans coupure ?",
    "Donnez-moi les tendances DevOps actuelles selon le dernier rapport DORA."
]

def run_evaluation(questions, category):
    print(f"\n{'='*50}\nÉVALUATION : QUESTIONS {category.upper()}\n{'='*50}")
    
    for i, q in enumerate(questions, 1):
        start_time = time.time()
        result = app.invoke({"question": q, "documents": [], "web_search_required": False})
        end_time = time.time()
        
        search_used = "OUI" if result.get("web_search_required") else "NON"
        print(f"\n[Q{i}] {q}")
        print(f"-> Temps : {end_time - start_time:.2f}s | Recherche Web déclenchée : {search_used}")
        print(f"-> Réponse : {result['generation']}")

if __name__ == "__main__":
    run_evaluation(questions_simples, "Simples")
    run_evaluation(questions_complexes, "Complexes")