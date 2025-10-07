# Language Planner - Structured plan generation with domain-driven constraints

import openai
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

from src.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE, TIMEOUT_SECONDS
from src.pddl_ir import DomainIR, ProblemIR, PlanStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguagePlanner:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

    def create_plan(
        self,
        domain_ir: DomainIR,
        problem_ir: ProblemIR,
        problem_description: str,
        conceptual_model: Optional[Dict[str, Any]] = None,
        problem_pddl: Optional[str] = None,
        feedback: Optional[str] = None,
        domain_pddl: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Use LLM to produce a structured plan constrained by the parsed domain/problem.

        Args:
            domain_ir: Parsed domain intermediate representation
            problem_ir: Parsed problem intermediate representation
            problem_description: Natural language description of the problem
            conceptual_model: Optional conceptual understanding from ConceptModeler
            problem_pddl: Optional raw problem PDDL string for context

        Returns:
            Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
                - natural_language_plan: short textual rationale
                - metadata: reasoning steps and auxiliary info
                - structured_plan: [{"name": str, "args": [str, ...]}, ...]
        """
        try:
            allowed_actions = [
                {"name": a.name, "arity": len(a.parameters)}
                for a in domain_ir.actions.values()
            ]
            allowed_objects = list(problem_ir.objects)

            context_parts: List[str] = []
            if conceptual_model is not None:
                context_parts.append("CONCEPTUAL OVERVIEW:\n" + json.dumps(conceptual_model, indent=2))
            context_parts.append("DOMAIN SYMBOLS (derived from parsed domain):\n" + json.dumps(allowed_actions, indent=2))
            context_parts.append("OBJECTS (from parsed problem):\n" + json.dumps(allowed_objects, indent=2))
            if problem_pddl is not None:
                context_parts.append("PROBLEM PDDL:\n" + problem_pddl)
            if domain_pddl is not None:
                context_parts.append("DOMAIN PDDL:\n" + domain_pddl)
            if feedback is not None:
                context_parts.append("FEEDBACK:\n" + feedback)

            # Action schemas (with preconditions/effects) derived from parsed domain
            action_specs = []
            for name, schema in domain_ir.actions.items():
                action_specs.append({
                    "name": name,
                    "arity": len(schema.parameters),
                    "parameters": schema.parameters,
                    "preconditions": schema.preconditions,
                    "effects": schema.effects
                })
            context_parts.append("ACTIONS (schemas):\n" + json.dumps(action_specs, indent=2))

            # Parsed INIT/GOAL (symbolic)
            context_parts.append("INIT (parsed):\n" + json.dumps(problem_ir.init, indent=2))
            context_parts.append("GOAL (parsed):\n" + json.dumps(problem_ir.goal, indent=2))

            # Generic, domain-agnostic guidance to reduce trivial or invalid plans
            context_parts.append(
                "PLANNING GUIDANCE:\n"
                "- Simulate preconditions and effects at each step using ACTIONS (schemas) and INIT.\n"
                "- Produce 3-6 steps; avoid 0-1 step plans unless the goal already holds.\n"
                "- Each step's preconditions must hold in the current symbolic state; update the state with the effects.\n"
                "- The final symbolic state must entail the goal predicates.\n"
                "- Use only allowed actions/objects; respect action arities exactly."
            )

            context_blob = "\n\n".join(context_parts)

            prompt = f"""You are planning strictly within a given symbolic domain. You must only use allowed action names and problem objects.

GOALS:
- Produce a valid action sequence that can be checked by a PDDL validator.
- Use only the allowed actions and objects provided below.
- Do not invent any new action names or objects.
- Respect action arities exactly.

CONSTRAINTS:
- Allowed actions: each action has a name and an arity (number of parameters).
- Allowed objects: every argument must be one of these strings.
- Output must be strict JSON following the required schema.

{context_blob}

Required JSON schema (no extra keys, no comments):
{{
  "reasoning_steps": [
    "Step 1 ...",
    "Step 2 ..."
  ],
  "plan": [
    {{"name": "ACTION_NAME", "args": ["obj1", "obj2"]}}
  ],
  "rationale": [
    "why this plan works in brief bullet points"
  ]
}}

Rules:
- Plan length MUST be between 3 and 6 steps unless the goal already holds in INIT.
- name must be one of the allowed action names.
- args length must equal the action arity.
- every arg must be one of the allowed objects.
- Provide a concise state_trace describing for each step which preconditions are satisfied and which predicates are added/deleted.
- Keep the plan minimal and consistent with the problem.
"""

            params = {
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            # o1/o-series models do not support 'max_tokens' on chat.completions
            if not str(OPENAI_MODEL).lower().startswith("o1"):
                params["temperature"] = 0.2
                params["max_tokens"] = 2000
            response = self.client.chat.completions.create(
                **params,
                timeout=TIMEOUT_SECONDS
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON block if wrapped in markdown
            json_start = content.find("```json")
            if json_start != -1:
                json_start += len("```json")
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
                else:
                    content = content[json_start:].strip()
            else:
                brace_start = content.find("{")
                if brace_start != -1:
                    content = content[brace_start:].strip()

            data = json.loads(content)

            # Validate structure
            reasoning_steps = data.get("reasoning_steps", [])
            rationale = data.get("rationale", [])
            plan_items = data.get("plan", [])

            if not isinstance(plan_items, list):
                raise ValueError("Invalid JSON: 'plan' must be a list")

            structured_plan: List[Dict[str, Any]] = []
            for i, it in enumerate(plan_items):
                if not isinstance(it, dict):
                    raise ValueError(f"Invalid plan item at index {i}: not an object")
                name = it.get("name")
                args = it.get("args", [])
                if not isinstance(name, str):
                    raise ValueError(f"Invalid plan item at index {i}: 'name' must be string")
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    raise ValueError(f"Invalid plan item at index {i}: 'args' must be list[str]")
                structured_plan.append({"name": name, "args": args})

            natural_language_plan = " ".join(rationale) if isinstance(rationale, list) else str(rationale)
            metadata: Dict[str, Any] = {
                "reasoning_steps": reasoning_steps,
                "rationale": rationale,
                "allowed_actions": allowed_actions,
                "allowed_objects": allowed_objects
            }

            logger.info("Successfully created structured plan")
            return natural_language_plan, metadata, structured_plan

        except Exception as e:
            logger.error(f"Error creating structured plan: {str(e)}")
            raise
