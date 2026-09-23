import datetime
import json
import logging
import os
import re

import pandas as pd
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import AsyncOpenAI, OpenAI

from src.config.config import api_keys_config, app_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_DATASET_PATH = "tests/evaluation/test_dataset.json"

# ============================================================================
# Wrapper Mistral pour DeepEval
# ============================================================================
class MistralEvaluatorLLM(DeepEvalBaseLLM):
    """Wrapper pour utiliser l'API Mistral AI comme LLM d'évaluation dans DeepEval."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=60.0
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=60.0
        )

    def get_model_name(self) -> str:
        """Méthode pour identifier le modèle."""
        return self.model_name

    def load_model(self):
        return self.client

    def _clean_json_output(self, text: str) -> str:
        """Nettoie la réponse du LLM pour garantir un JSON valide à DeepEval."""
        # 1. Extrait le contenu si Mistral entoure le JSON de balises ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        
        # 2. Cherche le premier '{' et le dernier '}' pour ignorer le texte superflu
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        return text

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,            
            #response_format={"type": "json_object"}           
        )

        raw_output = chat_completion.choices[0].message.content
        return self._clean_json_output(raw_output)

    async def a_generate(self, prompt: str) -> str:
        chat_completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,            
            #response_format={"type": "json_object"}
        )

        raw_output = chat_completion.choices[0].message.content
        return self._clean_json_output(raw_output)


# ============================================================================
# Chargement du dataset
# ============================================================================
def load_test_cases(dataset_path: str = TEST_DATASET_PATH) -> list[LLMTestCase]:
    logger.info(f"Chargement du jeu de test depuis {dataset_path}...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = []
    
    for question, answer, context, ref in zip(
        test_data["questions"],
        test_data["answers"],
        test_data["contexts"],
        test_data["ground_truths"]
    ):
        contexts_list = [context] if isinstance(context, str) else context
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts_list,
            expected_output=ref
        )
        test_cases.append(test_case)

    logger.info(f"{len(test_cases)} cas de test chargés et validés avec succès.")
    return test_cases

# ============================================================================
# Exécution des évaluations
# ============================================================================
def run_evaluation(test_cases: list[LLMTestCase], mistral_llm: MistralEvaluatorLLM) -> list[dict]:
    """Évalue chaque cas de test et retourne une liste de dictionnaires structurés."""

    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=mistral_llm)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=mistral_llm)    
    contextual_precision_metric = ContextualPrecisionMetric(threshold=0.5, model=mistral_llm)
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5, model=mistral_llm)

    results = []

    logger.info("Démarrage de l'évaluation...")

    for idx, test_case in enumerate(test_cases, 1):
        logger.info(f"Évaluation du cas {idx}/{len(test_cases)}...")

        # Exécution des mesures
        faithfulness_metric.measure(test_case)
        relevancy_metric.measure(test_case)
        contextual_precision_metric.measure(test_case)
        contextual_recall_metric.measure(test_case)

        # Structure claire et lisible par échantillon
        sample_result = {
            "id": idx,
            "question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "scores": {
                "faithfulness": faithfulness_metric.score,
                "relevancy": relevancy_metric.score,
                "contextual_precision": contextual_precision_metric.score,
                "contextual_recall": contextual_recall_metric.score
            },
            "reasons": {
                "faithfulness": faithfulness_metric.reason,
                "relevancy": relevancy_metric.reason,
                "contextual_precision": contextual_precision_metric.reason,
                "contextual_recall": contextual_recall_metric.reason
            },
        }

        results.append(sample_result)

    return results


def save_results_to_json(results: list[dict], output_path: str = "tests/evaluation/results.json"):
    """Sauvegarde les résultats détaillés dans un fichier JSON mis en forme."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Résultats sauvegardés avec succès dans : {output_path}")


def display_summary_table(results: list[dict]):
    """Affiche un tableau récapitulatif compact des scores dans la console."""
    summary_data = []

    for r in results:
        summary_data.append(
            {
                "ID": r["id"],
                "Question": (
                    r["question"][:40] + "..."
                    if len(r["question"]) > 40
                    else r["question"]
                ),
                "Faithfulness": r["scores"]["faithfulness"],
                "Relevancy": r["scores"]["relevancy"],
                "contextual_precision": r["scores"]["contextual_precision"],
                "contextual_recall": r["scores"]["contextual_recall"]
            }
        )

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 60)
    print("           RÉSUMÉ DES SCORES D'ÉVALUATION")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60 + "\n")

# ============================================================================
# Point d'entrée principal
# ============================================================================
if __name__ == "__main__":
    dataset_path = TEST_DATASET_PATH
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    mistral_evaluator = MistralEvaluatorLLM(
        model_name=app_config.models.evaluation_chat_model,
        api_key=api_keys_config.mistral.get_secret_value(),
    )

    test_cases = load_test_cases(dataset_path)
    results = run_evaluation(test_cases, mistral_evaluator)

    json_output_path = os.path.join(
        os.path.dirname(dataset_path), f"evaluation_results_{datetime.datetime.now(tz=datetime.UTC).strftime("%d-%m-%Y_%H%M%S")}.json"
    )
    save_results_to_json(results, output_path=json_output_path)

    display_summary_table(results)