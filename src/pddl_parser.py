from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from src.pddl_ir import DomainIR, ProblemIR, ActionSchema


def _tokenize(s: str) -> List[str]:
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]


def _parse(tokens: List[str]) -> Any:
    def read(i: int) -> Tuple[Any, int]:
        if tokens[i] != "(":
            return tokens[i], i + 1
        i += 1
        out = []
        while i < len(tokens) and tokens[i] != ")":
            node, i = read(i)
            out.append(node)
        return out, i + 1
    node, j = read(0)
    return node


def _is_kw(x: Any, kw: str) -> bool:
    return isinstance(x, str) and x.lower() == kw


def _sexpr_find_blocks(tree: Any, head_kw: str) -> List[Any]:
    blocks = []
    if isinstance(tree, list) and tree:
        if isinstance(tree[0], str) and tree[0].lower() == head_kw:
            blocks.append(tree)
        for ch in tree:
            blocks.extend(_sexpr_find_blocks(ch, head_kw))
    return blocks


def _arity(pred_node: List[Any]) -> int:
    # (p ?x ?y) -> 2
    return max(0, len(pred_node) - 1)


def _extract_domain_name(tree: Any) -> str:
    # (define (domain NAME) ...)
    if not (isinstance(tree, list) and tree and _is_kw(tree[0], "define")):
        return "unknown"
    for node in tree[1:]:
        if isinstance(node, list) and len(node) >= 2 and _is_kw(node[0], "domain"):
            if isinstance(node[1], str):
                return node[1]
    return "unknown"


def parse_domain(domain_pddl: str) -> DomainIR:
    tokens = _tokenize(domain_pddl)
    root = _parse(tokens)

    # domain name
    name = _extract_domain_name(root)

    # predicates
    predicates: Dict[str, int] = {}
    for blk in _sexpr_find_blocks(root, ":predicates"):
        # (:predicates (p ?x) (q ?x ?y) ...)
        for pred in blk[1:]:
            if isinstance(pred, list) and pred:
                head = pred[0]
                if isinstance(head, str):
                    predicates[head] = _arity(pred)

    # actions
    actions: Dict[str, ActionSchema] = {}
    for blk in _sexpr_find_blocks(root, ":action"):
        # (:action name :parameters (...) :precondition (...) :effect (...))
        if len(blk) >= 2 and isinstance(blk[1], str):
            act_name = blk[1]
            params: List[str] = []
            precond: Any = None
            effects: Any = None

            # scan tags inside this action block
            i = 2
            while i < len(blk):
                node = blk[i]
                if _is_kw(node, ":parameters") and i + 1 < len(blk):
                    if isinstance(blk[i + 1], list):
                        params = [p for p in blk[i + 1] if isinstance(p, str)]
                    i += 2
                    continue
                if _is_kw(node, ":precondition") and i + 1 < len(blk):
                    precond = blk[i + 1]
                    i += 2
                    continue
                if _is_kw(node, ":effect") and i + 1 < len(blk):
                    effects = blk[i + 1]
                    i += 2
                    continue
                i += 1

            actions[act_name] = ActionSchema(
                name=act_name,
                parameters=params,
                preconditions=precond,
                effects=effects
            )

    return DomainIR(name=name, predicates=predicates, actions=actions)


def parse_problem(problem_pddl: str, domain: DomainIR | None = None) -> ProblemIR:
    tokens = _tokenize(problem_pddl)
    root = _parse(tokens)

    objects: List[str] = []
    for blk in _sexpr_find_blocks(root, ":objects"):
        # (:objects a b c)
        for sym in blk[1:]:
            if isinstance(sym, str):
                objects.append(sym)

    # init/goal raw subtrees (keep original structure for downstream if needed)
    init_blocks = _sexpr_find_blocks(root, ":init")
    goal_blocks = _sexpr_find_blocks(root, ":goal")
    init_tree = init_blocks[0][1:] if init_blocks else []
    goal_tree = goal_blocks[0][1:] if goal_blocks else []

    return ProblemIR(objects=objects, init=init_tree, goal=goal_tree)
