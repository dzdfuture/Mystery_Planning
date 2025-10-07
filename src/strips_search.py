from collections import deque
from typing import Any, Dict, Iterable, List, Set, Tuple
from src.pddl_ir import DomainIR, ProblemIR, ActionSchema


Literal = Tuple[str, Tuple[str, ...]]  # ("pred", ("a","b",...))


def _is_list(x: Any) -> bool:
    return isinstance(x, list)


def _is_sym(x: Any) -> bool:
    return isinstance(x, str)


def _flatten_and(node: Any) -> List[Any]:
    if node is None:
        return []
    if _is_list(node) and node and _is_sym(node[0]) and node[0].lower() == "and":
        out: List[Any] = []
        for ch in node[1:]:
            out.extend(_flatten_and(ch))
        return out
    return [node]


def _literal_from_term(term: Any) -> Tuple[Literal, bool]:
    if not _is_list(term) or not term:
        raise ValueError(f"Invalid literal term: {term}")
    if _is_sym(term[0]) and term[0].lower() == "not":
        base = term[1]
        if not _is_list(base) or not base or not _is_sym(base[0]):
            raise ValueError(f"Invalid negated literal: {term}")
        pred = base[0]
        args = tuple(a for a in base[1:] if _is_sym(a))
        return (pred, args), True
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
            if lit in facts:
                facts.remove(lit)
    return facts


def _goal_holds(state: Set[Literal], goal_tree: Any) -> bool:
    for t in _flatten_and(goal_tree):
        if not _is_list(t):
            continue
        lit, is_neg = _literal_from_term(t)
        if is_neg:
            if lit in state:
                return False
        else:
            if lit not in state:
                return False
    return True


def _ground(lit: Literal, var_map: Dict[str, str]) -> Literal:
    pred, args = lit
    return (pred, tuple(var_map.get(a, a) for a in args))


def _preconds(schema: ActionSchema) -> List[Tuple[Literal, bool]]:
    out: List[Tuple[Literal, bool]] = []
    for t in _flatten_and(schema.preconditions):
        if not _is_list(t):
            continue
        lit, is_neg = _literal_from_term(t)
        out.append((lit, is_neg))
    return out


def _effects(schema: ActionSchema) -> Tuple[List[Literal], List[Literal]]:
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


def _applicable(state: Set[Literal], schema: ActionSchema, args: Tuple[str, ...]) -> bool:
    if len(args) != len(schema.parameters):
        return False
    var_map = dict(zip(schema.parameters, args))
    for lit, is_neg in _preconds(schema):
        g = _ground(lit, var_map)
        if is_neg:
            if g in state:
                return False
        else:
            if g not in state:
                return False
    return True


def _succ(state: Set[Literal], schema: ActionSchema, args: Tuple[str, ...]) -> Set[Literal]:
    var_map = dict(zip(schema.parameters, args))
    add_list, del_list = _effects(schema)
    new_state = set(state)
    for lit in del_list:
        g = _ground(lit, var_map)
        if g in new_state:
            new_state.remove(g)
    for lit in add_list:
        g = _ground(lit, var_map)
        new_state.add(g)
    return new_state


def _all_groundings(objects: List[str], arity: int) -> Iterable[Tuple[str, ...]]:
    if arity == 0:
        yield tuple()
        return
    # Cartesian product objects^arity
    # Small domains (e.g., a,b,c,d) keep this manageable
    def rec(prefix: List[str], k: int):
        if k == 0:
            yield tuple(prefix)
            return
        for o in objects:
            prefix.append(o)
            yield from rec(prefix, k - 1)
            prefix.pop()
    yield from rec([], arity)


def plan_via_search(domain: DomainIR, problem: ProblemIR, max_depth: int = 6, min_depth: int = 3) -> List[Dict[str, Any]]:
    """
    Simple breadth-first forward search planner using parsed STRIPS-like semantics.
    Returns a structured plan [{"name": str, "args": [str,...]}, ...] or [] if none found within depth.
    Enforces a minimum depth by not checking goal until the current layer reaches min_depth (unless goal holds initially).
    """
    # Build initial and goal
    init_tree = ["and"] + problem.init if not (_is_list(problem.init) and problem.init and problem.init[0] == "and") else problem.init
    goal_tree = ["and"] + problem.goal if not (_is_list(problem.goal) and problem.goal and problem.goal[0] == "and") else problem.goal

    init_state = frozenset(_state_from_init(init_tree))
    if _goal_holds(set(init_state), goal_tree):
        return []  # already satisfied

    # Precompute ground action instances (name, args, schema)
    ground_acts: List[Tuple[str, Tuple[str, ...], ActionSchema]] = []
    objs = list(problem.objects)
    for name, schema in domain.actions.items():
        arity = len(schema.parameters)
        for args in _all_groundings(objs, arity):
            ground_acts.append((name, args, schema))

    # BFS
    frontier: deque[frozenset[Literal]] = deque([init_state])
    parent: Dict[frozenset[Literal], Tuple[frozenset[Literal], str, Tuple[str, ...]]] = {}
    depth: Dict[frozenset[Literal], int] = {init_state: 0}
    visited: Set[frozenset[Literal]] = {init_state}

    while frontier:
        s = frontier.popleft()
        d = depth[s]
        # Only check goal when reaching min_depth or beyond
        if d >= min_depth and _goal_holds(set(s), goal_tree):
            # reconstruct
            plan_rev: List[Dict[str, Any]] = []
            cur = s
            while cur in parent:
                prev, aname, aargs = parent[cur]
                plan_rev.append({"name": aname, "args": list(aargs)})
                cur = prev
            plan_rev.reverse()
            return plan_rev

        if d == max_depth:
            continue

        for aname, aargs, schema in ground_acts:
            if not _applicable(set(s), schema, aargs):
                continue
            ns = frozenset(_succ(set(s), schema, aargs))
            if ns in visited:
                continue
            visited.add(ns)
            parent[ns] = (s, aname, aargs)
            depth[ns] = d + 1
            frontier.append(ns)

    return []  # not found within bounds
