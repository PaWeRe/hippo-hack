#!/usr/bin/env python3
"""
Compare LLM responses with and without RAG retrieval for medical questions.
This helps evaluate if the knowledge base improves answer quality.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_mistralai import ChatMistralAI
from src.agent.graph import retrieve_medical_knowledge


def test_rag_vs_no_rag():
    """Compare responses with and without RAG retrieval."""

    # Initialize LLM without tools
    api_key = os.getenv("API_KEY_MISTRAL")
    llm = ChatMistralAI(api_key=api_key, model="mistral-large-latest", temperature=0.3)

    medical_questions = [
        "What is cardiomegaly and what causes it?",
        "How does pneumonia appear on chest X-rays?",
        "What is pleural effusion and how is it treated?",
        "What are the symptoms of pneumothorax?",
        "How do you interpret a normal chest X-ray?",
    ]

    print("=" * 80)
    print("MEDICAL KNOWLEDGE COMPARISON: LLM vs LLM + RAG")
    print("=" * 80)

    for i, question in enumerate(medical_questions, 1):
        print(f"\n🔍 QUESTION {i}: {question}")
        print("-" * 60)

        # LLM alone response
        print("🤖 LLM ALONE:")
        llm_response = llm.invoke(f"As a medical AI, answer this question: {question}")
        print(
            llm_response.content[:300] + "..."
            if len(llm_response.content) > 300
            else llm_response.content
        )

        print("\n🧠 LLM + RAG KNOWLEDGE:")
        # Get knowledge from RAG using the retrieve_medical_knowledge function
        # This function returns a structured response with medical knowledge
        try:
            knowledge_response = retrieve_medical_knowledge({"query": question})
            knowledge_info = {
                "documents": [knowledge_response] if knowledge_response else []
            }
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            knowledge_info = {"documents": []}

        if knowledge_info["documents"]:
            # Create enhanced prompt with retrieved knowledge
            enhanced_prompt = f"""As a medical AI, answer this question using the provided medical knowledge context.

Question: {question}

Medical Knowledge Context:
{knowledge_info['documents'][0]}

Please provide a comprehensive answer based on both your training and the provided context."""

            rag_response = llm.invoke(enhanced_prompt)
            print(
                rag_response.content[:300] + "..."
                if len(rag_response.content) > 300
                else rag_response.content
            )
        else:
            print("No relevant knowledge found in database.")

        print("\n" + "=" * 60)

    print(f"\n📚 Knowledge Base Statistics:")
    print(f"Vector store contains medical information about common X-ray findings")
    print(f"Embeddings model: sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    test_rag_vs_no_rag()
