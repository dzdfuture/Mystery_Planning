from typing import Any, Dict, List, Union
from dataclasses import asdict
from src.pddl_ir import DomainIR, ProblemIR, PlanStep


class PDDLTranslator:
    """
    Translate a structured action sequence into VAL-compatible textual plan.
    - No hard-coding of domain symbols: all validation uses DomainIR/ProblemIR.
    - Output format matches existing VALInterface expectations:
      ["action(arg1,arg2)", "other()"] which VALInterface will convert to PDDL "(action arg1 arg2)".
    """

    @staticmethod
    def _to_plan_steps(steps: List[Union[PlanStep, Dict[str, Any]]]) -> List[PlanStep]:
        norm: List[PlanStep] = []
        for s in steps:
            if isinstance(s, PlanStep):
                norm.append(s)
            elif isinstance(s, dict):
                name = s.get("name")
                args = s.get("args", [])
                if not isinstance(name, str):
                    raise ValueError("Plan step missing valid 'name' string")
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    raise ValueError("Plan step 'args' must be a list of strings")
                norm.append(PlanStep(name=name, args=args))
            else:
                raise ValueError("Unsupported plan step type")
        return norm

    @staticmethod
    def _validate_steps(steps: List[PlanStep], domain_ir: DomainIR, problem_ir: ProblemIR) -> List[str]:
        errors: List[str] = []
        allowed_actions = set(domain_ir.actions.keys())
        allowed_objects = set(problem_ir.objects)

        for idx, step in enumerate(steps):
            # action existence
            if step.name not in allowed_actions:
                errors.append(f"step {idx}: unknown action '{step.name}' (allowed: {sorted(allowed_actions)})")
                continue

            schema = domain_ir.actions[step.name]
            # arity check by parameter count (types not used in PlanBench files)
            if len(step.args) != len(schema.parameters):
                errors.append(
                    f"step {idx}: arity mismatch for '{step.name}': "
                    f"expected {len(schema.parameters)}, got {len(step.args)}"
                )

            # object existence
            for a in step.args:
                if a not in allowed_objects:
                    errors.append(
                        f"step {idx}: argument '{a}' not in problem objects {sorted(allowed_objects)}"
                    )

        return errors

    @staticmethod
    def _format_for_val_interface(steps: List[PlanStep]) -> List[str]:
        plan_lines: List[str] = []
        for s in steps:
            if s.args:
                plan_lines.append(f"{s.name}({','.join(s.args)})")
            else:
                plan_lines.append(f"{s.name}()")
        return plan_lines

    def translate_to_pddl(
        self,
        plan_steps: List[Union[PlanStep, Dict[str, Any]]],
        domain_ir: DomainIR,
        problem_ir: ProblemIR,
        strict: bool = True,
    ) -> List[str]:
        """
        Convert structured steps into textual lines for VALInterface.

        Args:
            plan_steps: List of steps, either PlanStep or dicts with {"name": str, "args": [str,...]}
            domain_ir: Parsed domain IR
            problem_ir: Parsed problem IR
            strict: If True, raise on validation errors; if False, filter invalid steps

        Returns:
            List[str]: e.g., ["feast(b,c)", "succumb(b)"]
        """
        steps = self._to_plan_steps(plan_steps)
        errors = self._validate_steps(steps, domain_ir, problem_ir)
        if errors:
            if strict:
                raise ValueError("Plan validation failed: " + "; ".join(errors))
            # non-strict: drop invalid steps
            valid: List[PlanStep] = []
            allowed_actions = set(domain_ir.actions.keys())
            allowed_objects = set(problem_ir.objects)
            for s in steps:
                if s.name not in allowed_actions:
                    continue
                if len(s.args) != len(domain_ir.actions[s.name].parameters):
                    continue
                if any(a not in allowed_objects for a in s.args):
                    continue
                valid.append(s)
            steps = valid

        return self._format_for_val_interface(steps)
