# Concept World Modeler - Deep language understanding of PDDL domains via LLM

import openai
import json
import logging
import os
from src.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE, TIMEOUT_SECONDS
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptModeler:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

    def model_domain(self, domain_file_path: str) -> Dict[str, Any]:
        """
        Use LLM to deep understand the domain rules and create a conceptual model
        without hard-coded mappings or pattern matching.

        Args:
            domain_file_path (str): Path to the PDDL domain file

        Returns:
            Dict[str, Any]: Conceptual understanding model
        """
        try:
            # Read the domain file
            if not os.path.exists(domain_file_path):
                raise FileNotFoundError(f"Domain file not found: {domain_file_path}")
            with open(domain_file_path, 'r') as f:
                domain_content = f.read()

            # Construct prompt for deep domain understanding
            prompt = f"""
You are an abstract reasoning specialist focused on mathematical and logical relationships, NOT physical concepts.

CRITICAL COGNITIVE INSTRUCTION:
- DO NOT assume any physical world interpretations
- Treat "craves" as abstract mathematical relations, NOT emotional states
- Focus on predicate logic transformations, NOT spatial relationships
- Understand actions as axiom transformations, NOT physical operations

Analyze this PDDL domain specification:

{domain_content}

Your task is to understand this domain at the abstract predicate level:

1. What abstract objects exist in this logical system
2. What mathematical relationships (predicates) can exist between objects
3. What logical transformations (actions) are available and their precise effects
4. What logical goal patterns this domain expresses
5. How this domain works as a system of predicate logic operations

PROHIBITED: Any mention of physical concepts like "stacking", "placing", "moving", "picking up"
REQUIRED: Treat all predicates and actions as abstract mathematical/logical operations

Provide your analysis in JSON format:
{{
    "domain_name": "name of the domain",
    "object_types": ["list", "of", "abstract", "objects"],
    "logical_relationships": "pure abstract description of predicate relationships",
    "logical_transformations": {{
        "action_name": "precise description of predicate changes from preconditions to effects"
    }},
    "abstract_goal_patterns": "what the logical achievement looks like",
    "logical_overview": "how this domain operates as pure abstract logic"
}}

Focus purely on the logical/axiomatic structure. No physical interpretations allowed.
"""

            # Call LLM API
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use deterministic output for consistent analysis
                max_tokens=2000,
                timeout=TIMEOUT_SECONDS
            )

            # Parse JSON response
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

            try:
                conceptual_model = json.loads(response_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed after cleanup. Response content: {response_content}")
                logger.error(f"JSON error: {json_err}")
                raise ValueError(f"Invalid JSON response from LLM after cleanup: {response_content[:200]}...")

            logger.info(f"Successfully created conceptual model for domain: {conceptual_model.get('domain_name', 'unknown')}")

            return conceptual_model

        except Exception as e:
            logger.error(f"Error modeling domain: {str(e)}")
            raise
