import os
import time
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ==========================================
# 1. PRÉPARATION DE LA BASE VECTORIELLE
# ==========================================
def setup_retriever():
    loader = TextLoader("data/cloud_devops.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
web_search_tool = TavilySearchResults(max_results=2)

# ==========================================
# 2. DÉFINITION DE L'ÉTAT DU GRAPHE
# ==========================================
class AgentState(TypedDict):
    question: str
    documents: List[str]
    web_search_required: bool
    generation: str

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Les documents sont pertinents pour la question, 'yes' ou 'no'")

# ==========================================
# 3. DÉFINITION DES NŒUDS (NODES)
# ==========================================
def retrieve_node(state: AgentState):
    """Récupère les documents de la base locale."""
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": [doc.page_content for doc in docs]}

def grade_node(state: AgentState):
    """Évalue la pertinence des documents récupérés vis-à-vis de la question."""
    question = state["question"]
    documents = state["documents"]
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system_prompt = "Vous êtes un évaluateur. Vérifiez si les documents contiennent les mots-clés ou le sens pour répondre à la question. Répondez 'yes' ou 'no'."
    
    context = "\n---\n".join(documents)
    response = structured_llm_grader.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nDocuments:\n{context}")
    ])
    
    if response.binary_score.lower() == "yes":
        return {"web_search_required": False}
    else:
        return {"web_search_required": True}

def web_search_node(state: AgentState):
    """Effectue une recherche Web si les documents locaux sont insuffisants."""
    question = state["question"]
    documents = state["documents"]
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(f"Résultats Web:\n{web_results}")
    
    return {"documents": documents}

def generate_node(state: AgentState):
    """Génère la réponse finale."""
    question = state["question"]
    documents = state["documents"]
    
    context = "\n---\n".join(documents)
    prompt = f"Répondez à la question en utilisant le contexte suivant. Si vous ne savez pas, dites-le.\n\nContexte: {context}\n\nQuestion: {question}"
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# ==========================================
# 4. CONSTRUCTION ET ROUTAGE DU GRAPHE
# ==========================================
def decide_to_generate(state: AgentState):
    if state["web_search_required"]:
        return "web_search_node"
    return "generate_node"

workflow = StateGraph(AgentState)

workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_node", grade_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_node", generate_node)

workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "grade_node")
workflow.add_conditional_edges("grade_node", decide_to_generate)
workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("generate_node", END)

app = workflow.compile()

# Visualisation (Sauvegarde de l'architecture)
try:
    with open("graph_architecture.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
except Exception as e:
    print("Impossible de générer l'image PNG.")