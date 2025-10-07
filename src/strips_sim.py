from typing import Any, Dict, List, Set, Tuple
from dataclasses import dataclass
from src.pddl_ir import DomainIR, ProblemIR, ActionSchema, PlanStep


Literal = Tuple[str, Tuple[str, ...]]  # ("pred", ("a","b",...))


def _is_list(x: Any) -> bool:
    return isinstance(x, list)


def _is_sym(x: Any) -> bool:
    return isinstance(x, str)


def _flatten_and(node: Any) -> List[Any]:
    # Turn (and a b (and c d)) into [a,b,c,d]; single term -> [term]; None -> []
    if node is None:
        return []
    if _is_list(node) and node and _is_sym(node[0]) and node[0].lower() == "and":
        out: List[Any] = []
        for ch in node[1:]:
            out.extend(_flatten_and(ch))
        return out
    return [node]


def _literal_from_term(term: Any) -> Tuple[Literal, bool]:
    # Return ((pred, (args...)), is_neg)
    if not _is_list(term) or not term:
        raise ValueError(f"Invalid literal term: {term}")
    if _is_sym(term[0]) and term[0].lower() == "not":
        base = term[1]
        if not _is_list(base) or not base or not _is_sym(base[0]):
            raise ValueError(f"Invalid negated literal: {term}")
        pred = base[0]
        args = tuple(a for a in base[1:] if _is_sym(a))
        return (pred, args), True
    # positive
    if not _is_sym(term[0]):
        raise ValueError(f"Invalid literal head: {term}")
    pred = term[0]
    args = tuple(a for a in term[1:] if _is_sym(a))
    return (pred, args), False


def _state_from_init(init_tree: Any) -> Set[Literal]:
    facts: Set[Literal] = set()
    for t in _flatten_and(init_tree):
        if not _is_list(t):
            continue
        lit, is_neg = _literal_from_term(t)
        if not is_neg:
            facts.add(lit)
        else:
            # ignore explicit negatives in init (rare)
            if lit in facts:
                facts.remove(lit)
    return facts


def _goal_holds(state: Set[Literal], goal_tree: Any) -> bool:
    # Treat goal as conjunction; negative literal must be absent
    for t in _flatten_and(goal_tree):
        if not _is_list(t):
            # tolerate non-literal (shouldn't happen)
            continue
        lit, is_neg = _literal_from_term(t)
        if is_neg:
            if lit in state:
                return False
        else:
            if lit not in state:
                return False
    return True


def _ground_literal(lit: Literal, var_map: Dict[str, str]) -> Literal:
    pred, args = lit
    grounded = tuple(var_map.get(a, a) for a in args)
    return (pred, grounded)


def _extract_preconds(schema: ActionSchema) -> List[Tuple[Literal, bool]]:
    out: List[Tuple[Literal, bool]] = []
    for t in _flatten_and(schema.preconditions):
        if not _is_list(t):
            continue
        lit, is_neg = _literal_from_term(t)
        out.append((lit, is_neg))
    return out


def _extract_effects(schema: ActionSchema) -> Tuple[List[Literal], List[Literal]]:
    add_list: List[Literal] = []
    del_list: List[Literal] = []
    for t in _flatten_and(schema.effects):
        if not _is_list(t):
            continue
        lit, is_neg = _literal_from_term(t)
        if is_neg:
            del_list.append(lit)
        else:
            add_list.append(lit)
    return add_list, del_list


def _check_preconds(state: Set[Literal], preconds: List[Tuple[Literal, bool]], var_map: Dict[str, str]) -> List[str]:
    errs: List[str] = []
    for lit, is_neg in preconds:
        g = _ground_literal(lit, var_map)
        if is_neg:
            if g in state:
                errs.append(f"negated precondition violated: {g}")
        else:
            if g not in state:
                errs.append(f"missing precondition: {g}")
    return errs


def _apply_effects(state: Set[Literal], add_list: List[Literal], del_list: List[Literal], var_map: Dict[str, str]) -> Set[Literal]:
    new_state = set(state)
    for lit in del_list:
        g = _ground_literal(lit, var_map)
        if g in new_state:
            new_state.remove(g)
    for lit in add_list:
        g = _ground_literal(lit, var_map)
        new_state.add(g)
    return new_state


def simulate_plan(domain_ir: DomainIR, problem_ir: ProblemIR, steps: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Symbolically simulate a structured plan (list of {"name":str,"args":[...]}) under STRIPS-like semantics.
    Returns (ok, report). Report contains the first failure reason or success summary.
    """
    # Build initial state
    # problem_ir.init is a list of terms (or nested and)
    init_tree = ["and"] + problem_ir.init if not (_is_list(problem_ir.init) and problem_ir.init and problem_ir.init[0] == "and") else problem_ir.init
    goal_tree = ["and"] + problem_ir.goal if not (_is_list(problem_ir.goal) and problem_ir.goal and problem_ir.goal[0] == "and") else problem_ir.goal

    state = _state_from_init(init_tree)
    # Quick success check (goal already holds)
    if _goal_holds(state, goal_tree):
        return True, "goal already holds in initial state"

    for idx, it in enumerate(steps):
        name = it.get("name")
        args = it.get("args", [])
        if name not in domain_ir.actions:
            return False, f"step {idx+1}: unknown action '{name}'"
        schema = domain_ir.actions[name]
        if len(args) != len(schema.parameters):
            return False, f"step {idx+1}: arity mismatch for '{name}': expected {len(schema.parameters)}, got {len(args)}"
        # build var mapping
        var_map = {}
        for v, a in zip(schema.parameters, args):
            var_map[v] = a

        preconds = _extract_preconds(schema)
        missing = _check_preconds(state, preconds, var_map)
        if missing:
            return False, f"step {idx+1}: preconditions not satisfied for '{name}': " + "; ".join(missing)

        add_list, del_list = _extract_effects(schema)
        state = _apply_effects(state, add_list, del_list, var_map)

    if _goal_holds(state, goal_tree):
        return True, "goal satisfied after executing plan"
    return False, "plan finished but goal not satisfied"
