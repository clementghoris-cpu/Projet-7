import os
import json
import logging
from datasets import Dataset
import torch
from google import genai
from ragas import evaluate
from ragas.run_config import RunConfig
from openai import OpenAI
from ragas.llms import llm_factory, LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from mistralai.client import Mistral
from src.rag.rag_chain import RAGChainManager
from src.config.config import api_keys_config, app_config
from langchain_mistralai import ChatMistralAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_DATASET_PATH = "tests/evaluation/test_dataset.json"

def run_ragas_evaluation(dataset_path: str = TEST_DATASET_PATH):
    """Exécute l'évaluation automatisée du système RAG localement avec Mistral et Ragas."""
    logging.info(f"Chargement du jeu de test annoté depuis {dataset_path}...")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    rag_manager = RAGChainManager()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    logging.info("Génération des réponses RAG pour chaque question du jeu de test...")
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        result = rag_manager.answer_question(q)
        retrieved_chunks = rag_manager._retrieve_relevant_chunks(q)
        chunk_texts = [c.get("text", "") for c in retrieved_chunks] if retrieved_chunks else ["Aucun contexte."]

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(chunk_texts)
        ground_truths.append(gt)

    # Structuration au format Dataset requise par Ragas
    ragas_dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    ragas_dataset = Dataset.from_dict(ragas_dataset_dict)

    logging.info("Configuration du juge Ragas local via Google...")
    
    eval_llm = ChatMistralAI(
        model_name=app_config.models.evaluation_chat_model,
        api_key=api_keys_config.mistral.get_secret_value(),
        max_tokens=4096,
        temperature=0.0
    )

    run_config = RunConfig(timeout=600, max_retries=3, max_wait=60)
    llm = LangchainLLMWrapper(eval_llm, run_config=run_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}

    base_embeddings = HuggingFaceEmbeddings(
            model_name = app_config.models.embeddings_model,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs
    )

    embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    logging.info("Calcul des métriques Ragas en local...")
    try:
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings, ),
            AnswerCorrectness(llm=llm, embeddings=embeddings)
        ]            

        results = evaluate(
            dataset=ragas_dataset,
            metrics= metrics,
            batch_size=1
        )
        
        print("\n" + "="*50)
        print("         RÉSULTATS DE L'ÉVALUATION RAGAS (LOCAL)")
        print("="*50)
        print(results)
        print("="*50 + "\n")
        
        # Export des résultats
        df_results = results.to_pandas()
        df_results.to_csv("tests/evaluation/ragas_evaluation_results.csv", index=False)
        logging.info("Résultats de l'évaluation enregistrés dans 'tests/evaluation/ragas_evaluation_results.csv'.")
        
    except Exception as e:
        logging.exception(f"Erreur lors de l'exécution de Ragas : {e}")

if __name__ == "__main__":
    if not os.path.isabs(TEST_DATASET_PATH):
        dataset_path = os.path.join(os.getcwd(), TEST_DATASET_PATH)

    run_ragas_evaluation(dataset_path)