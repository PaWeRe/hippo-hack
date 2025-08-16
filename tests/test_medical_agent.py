#!/usr/bin/env python3
"""
Test script for the medical agent with runtime context and tool-based X-ray analysis.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import invoke_medical_agent


def test_xray_interpretation():
    """Test the X-ray interpretation workflow."""

    # Sample X-ray results
    sample_xray_results = {
        "Cardiomegaly": 0.85,
        "Pneumonia": 0.72,
        "Pleural_Effusion": 0.68,
        "Normal": 0.15,
        "Pneumothorax": 0.12,
    }

    # Test case 1: Patient asks for X-ray interpretation
    print("=" * 60)
    print("TEST 1: Patient asks for X-ray interpretation")
    print("=" * 60)

    result = invoke_medical_agent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Can you please help me interpret my X-ray findings?",
                }
            ],
            "xray_results": sample_xray_results,
            "patient_info": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "weight": 75,
                "height": 175,
                "blood_type": "A+",
                "hypertension": True,
            },
        }
    )

    print("Final response:", result["messages"][-1].content)
    print()

    # Test case 1b: Patient asks for recommendations
    print("=" * 60)
    print("TEST 1b: Patient asks for specific recommendations")
    print("=" * 60)

    result1b = invoke_medical_agent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Based on my X-ray findings, what specific steps should I take? I need clear recommendations.",
                }
            ],
            "xray_results": sample_xray_results,
            "patient_info": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "weight": 75,
                "height": 175,
                "blood_type": "A+",
                "hypertension": True,
            },
        }
    )

    print("Recommendations response:", result1b["messages"][-1].content)
    print()

    # Test case 2: Patient asks for explanation of specific condition
    print("=" * 60)
    print("TEST 2: Patient asks for explanation after seeing findings")
    print("=" * 60)

    # Simulate a follow-up conversation where patient wants more info
    result2 = invoke_medical_agent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Can you please help me interpret my X-ray findings?",
                },
                result["messages"][-1],  # Previous AI response
                {
                    "role": "user",
                    "content": "What does cardiomegaly mean exactly? Should I be worried?",
                },
            ],
            "xray_results": sample_xray_results,
            "patient_info": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "weight": 75,
                "height": 175,
                "blood_type": "A+",
                "hypertension": True,
            },
        }
    )

    print("Final response:", result2["messages"][-1].content)
    print()

    # Test case 3: General medical question (no X-ray interpretation needed)
    print("=" * 60)
    print("TEST 3: General medical question")
    print("=" * 60)

    result3 = invoke_medical_agent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the general symptoms of pneumonia?",
                }
            ],
            "xray_results": sample_xray_results,
            "patient_info": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "weight": 75,
                "height": 175,
                "blood_type": "A+",
                "hypertension": True,
            },
        }
    )

    print("Final response:", result3["messages"][-1].content)
    print()


if __name__ == "__main__":
    test_xray_interpretation()
