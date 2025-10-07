from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ActionSchema:
    name: str
    parameters: List[str]  # variable names as they appear in the domain (e.g., ?x, ?y)
    preconditions: Any = None  # raw s-expression subtree of :precondition
    effects: Any = None        # raw s-expression subtree of :effect


@dataclass
class DomainIR:
    name: str
    predicates: Dict[str, int]  # predicate name -> arity
    actions: Dict[str, ActionSchema]  # action name -> schema


@dataclass
class ProblemIR:
    objects: List[str]
    init: Any  # raw s-expression subtree of init
    goal: Any  # raw s-expression subtree of goal


@dataclass
class PlanStep:
    name: str
    args: List[str]
