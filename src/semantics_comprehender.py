# Semantics Comprehender - Understanding the functional essence of Mystery domain terms

import openai
import json
import logging
from src.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE, TIMEOUT_SECONDS
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticsComprehender:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

    def comprehend_semantics(self, mystery_domain_content: str, natural_language_plan: str,
                           conceptual_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM understand the functional essence of Mystery domain terms and their relationship
        to familiar concepts, without hard-coded mappings.

        Args:
            mystery_domain_content (str): Content of the Mystery domain PDDL file
            natural_language_plan (str): Natural language plan from LanguagePlanner
            conceptual_model (Dict[str, Any]): Conceptual understanding from ConceptModeler

        Returns:
            Dict[str, Any]: Semantic correspondences between terms and functions
        """
        try:
            prompt = f"""
You are an abstract reasoning specialist focused on predicate logic operations.

CRITICAL COGNITIVE FRAMEWORK:
- Treat ALL terms as pure predicate logic operations
- NEVER assume physical meaning for any term
- Understand "craves" as formal relation predicates, NOT desires
- Focus purely on logical transformations and axiom applications

DOMAIN ABSTRACT LOGIC:
{json.dumps(conceptual_model, indent=2)}

MYSTERY DOMAIN PDDL SPECIFICATION:
{mystery_domain_content}

ABSTRACT PLAN TO INTERPRET:
{natural_language_plan}

Your task is to interpret the Mystery domain at the pure logical level:

1. Identify each predicate as formal logical relations without physical interpretation
2. Understand each action as axiom transformation in predicate calculus
3. Map logical transformations to corresponding domain actions
4. Ensure complete detachment from physical concepts

PROHIBITED INTERPRETATIONS:
- NO mentions of spatial relationships, physical states, or object manipulations
- NO assumptions of emotional states or human concepts
- NO references to physical actions like moving, placing, stacking

REQUIRED ANALYSIS FRAMEWORK:
Think in terms of: predicates, logical transformations, axiom applications, logical state changes

Provide your semantic analysis in JSON format:
{{
    "predicate_interpretations": {{
        "mystery_predicate_1": {{
            "logical_definition": "pure abstract relation between objects",
            "logical_role": "how this predicate functions in logical transformations",
            "logical_interpretation": "how to understand this in predicate logic terms"
        }}
    }},
    "action_interpretations": {{
        "mystery_action_1": {{
            "logical_transformation": "what axiom transformation this represents",
            "precondition_logic": "logical preconditions as predicate formulas",
            "effect_logic": "logical effects as predicate changes"
        }}
    }},
    "logical_domain_overview": "Complete logical understanding of the domain as axiom system"
}}

REMAIN ENTIRELY WITHIN FORMAL LOGIC. NO PHYSICAL INTERPRETATIONS ALLOWED.
"""

            # Call LLM for semantic understanding
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low for consistent semantic analysis
                max_tokens=2500,
                timeout=TIMEOUT_SECONDS
            )

            # Parse response
            response_content = response.choices[0].message.content.strip()

            # Extract JSON from markdown code block
            json_start = response_content.find("```json")
            if json_start != -1:
                json_start += len("```json")
                json_end = response_content.find("```", json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
                else:
                    response_content = response_content[json_start:].strip()
            else:
                # Fallback: remove any leading text before opening brace
                brace_start = response_content.find("{")
                if brace_start != -1:
                    response_content = response_content[brace_start:].strip()
                else:
                    response_content = response_content.strip()

            semantic_understanding = json.loads(response_content)

            logger.info("Successfully comprehended semantics of Mystery domain")

            return semantic_understanding

        except Exception as e:
            logger.error(f"Error comprehending semantics: {str(e)}")
            raise
