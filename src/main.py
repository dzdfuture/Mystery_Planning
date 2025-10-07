# Main Entry Point - Unified experiment runner for LLM abstract reasoning validation system

import argparse
import json
import os
import logging
import time
from pathlib import Path
import concurrent.futures as futures
import threading
from src.config import *
from src.concept_modeler import ConceptModeler
from src.language_planner import LanguagePlanner
from src.semantics_comprehender import SemanticsComprehender
from src.pddl_translator import PDDLTranslator
from src.val_interface import VALInterface
from src.pddl_parser import parse_domain, parse_problem
from src.strips_sim import simulate_plan
from src.strips_search import plan_via_search

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self):
        self.concept_modeler = ConceptModeler() if ENABLE_CONCEPT else None
        self.language_planner = LanguagePlanner()
        self.semantics_comprehender = SemanticsComprehender() if ENABLE_SEMANTICS else None
        self.pddl_translator = PDDLTranslator()
        self.val_interface = VALInterface()

    def save_instance_result(self, instances_dir: Path, problem_id: str, result: dict):
        """
        Persist a single problem's result immediately.
        Safe for parallel execution (unique filename per problem_id).
        """
        instances_dir.mkdir(parents=True, exist_ok=True)
        out_path = instances_dir / f"{problem_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Instance result saved: {out_path}")


    def get_concept_model(self, domain_path: Path):
        if not ENABLE_CONCEPT:
            return {}
        key = str(domain_path)
        # init cache and lock
        if getattr(ExperimentRunner, "_concept_cache", None) is None:
            ExperimentRunner._concept_cache = {}
        if getattr(ExperimentRunner, "_concept_lock", None) is None:
            ExperimentRunner._concept_lock = threading.Lock()
        cache = ExperimentRunner._concept_cache
        lock: threading.Lock = ExperimentRunner._concept_lock
        # fast path
        if key in cache:
            return cache[key]
        # synchronize to avoid duplicate LLM calls
        with lock:
            if key in cache:
                return cache[key]
            cm = self.concept_modeler.model_domain(str(domain_path))
            cache[key] = cm
            return cm

    def build_problem_description(self, problem_pddl: str) -> str:
        if not ENABLE_PROBLEM_DESC_LLM:
            try:
                prob_ir = parse_problem(problem_pddl)
                return (
                    "OBJECTS:\n" + json.dumps(prob_ir.objects, indent=2) + "\n"
                    "INIT (symbolic):\n" + json.dumps(prob_ir.init, indent=2) + "\n"
                    "GOAL (symbolic):\n" + json.dumps(prob_ir.goal, indent=2)
                )
            except Exception:
                return "Use the provided PDDL problem directly as context; derive objects/init/goal from it."
        prompt = f"""
You are working in pure predicate logic. Convert this PDDL problem to a logical predicate description:

{problem_pddl}

DESCRIPTION REQUIREMENTS:
- Express objects as logical variables/constants
- Describe initial state as logical formulas (predicates)
- State goal state as predicate formulas to achieve
- Use abstract logical reasoning - avoid physical interpretations
- Focus on predicate transformations rather than object manipulations

REQUIRED FORMAT: Provide a clear logical description of:
1. Available objects in logical terms
2. Initial predicate state (what logical formulas hold initially)
3. Goal predicate state (what logical formulas must hold finally)

Frame everything in terms of predicate logic operations and state transformations.
"""
        response = self.language_planner.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            timeout=TIMEOUT_SECONDS
        )
        return response.choices[0].message.content.strip()

    def cleanup_legacy_result(self, domain: str, problem_id: str, model_name: str, instances_dir: Path):
        if not LEGACY_RESULTS_CLEANUP:
            return
        legacy_file = Path("results") / "mystery" / domain / "batch" / model_name / f"{problem_id}.json"
        try:
            if legacy_file.exists():
                target = instances_dir / f"{problem_id}.json"
                with open(legacy_file, "r") as f:
                    data = f.read()
                with open(target, "w") as f:
                    f.write(data)
                legacy_file.unlink()
                parent = legacy_file.parent
                try:
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        pp = parent.parent
                        if pp.name == "batch" and pp.exists() and not any(pp.iterdir()):
                            pp.rmdir()
                        ppp = pp.parent if pp else None
                        if ppp and ppp.name == domain and not any(ppp.iterdir()):
                            ppp.rmdir()
                except Exception:
                    pass
                logger.info(f"Migrated legacy result {legacy_file} -> {target}")
        except Exception as e:
            logger.warning(f"Legacy cleanup failed for {legacy_file}: {e}")
 
    def migrate_legacy_dir(self, domain: str, model_name: str, instances_dir: Path):
        if not LEGACY_RESULTS_CLEANUP:
            return
        legacy_dir = Path("results") / "mystery" / domain / "batch" / model_name
        if not legacy_dir.exists():
            return
        for jf in legacy_dir.glob("*.json"):
            try:
                target = instances_dir / jf.name
                if not target.exists():
                    target.write_text(jf.read_text())
                jf.unlink()
            except Exception as e:
                logger.warning(f"Failed to migrate legacy file {jf}: {e}")
        try:
            if legacy_dir.exists() and not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
                parent = legacy_dir.parent  # .../batch
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                gparent = parent.parent  # .../blocksworld
                if gparent.exists() and not any(gparent.iterdir()):
                    gparent.rmdir()
        except Exception:
            pass

    def start_legacy_watcher(self, domain: str, model_name: str, instances_dir: Path, interval: float = 2.0):
        if not LEGACY_RESULTS_CLEANUP:
            return
        def _worker():
            while True:
                try:
                    self.migrate_legacy_dir(domain, model_name, instances_dir)
                except Exception as e:
                    logger.warning(f"legacy watcher encountered error: {e}")
                time.sleep(interval)
        t = threading.Thread(target=_worker, daemon=True, name="legacy-migrator")
        t.start()
 
    def load_dataset_instance(self, domain: str, problem_id: str, problem_path_override: Path | None = None, domain_path_override: Path | None = None) -> tuple:
        """Load domain and problem files from PlanBench dataset with optional explicit overrides."""
        base = Path(PLANBENCH_INSTANCES_PATH) / domain

        # If explicit overrides are given, honor them
        if problem_path_override is not None:
            problem_path = Path(problem_path_override)
            if not problem_path.exists():
                raise FileNotFoundError(f"Problem file not found at override path: {problem_path}")
            if domain_path_override is not None:
                domain_path = Path(domain_path_override)
            else:
                # Heuristic: if 'mystery' in problem path, pick mystery domain, else standard generated_domain
                if "mystery" in problem_path.parts:
                    domain_path = base / "mystery" / "generated_domain.pddl"
                else:
                    domain_path = base / "generated_domain.pddl"
        else:
            # Auto-select mystery/standard domain as before
            mystery_problem = base / "mystery" / "generated_basic" / f"{problem_id}.pddl"
            standard_problem = base / f"{problem_id}.pddl"

            if mystery_problem.exists():
                domain_path = base / "mystery" / "generated_domain.pddl"
                problem_path = mystery_problem
            else:
                if not standard_problem.exists():
                    raise FileNotFoundError(f"Problem file not found: {mystery_problem} or {standard_problem}")
                domain_path = base / "generated_domain.pddl"
                problem_path = standard_problem

        if not domain_path.exists():
            raise FileNotFoundError(f"Domain file not found: {domain_path}")

        with open(domain_path, 'r') as f:
            domain_content = f.read()
        with open(problem_path, 'r') as f:
            problem_content = f.read()

        return domain_content, problem_content, domain_path

    def create_problem_description(self, problem_pddl: str) -> str:
        """Convert PDDL problem to logical predicate description for abstract reasoning."""
        prompt = f"""
You are working in pure predicate logic. Convert this PDDL problem to a logical predicate description:

{problem_pddl}

DESCRIPTION REQUIREMENTS:
- Express objects as logical variables/constants
- Describe initial state as logical formulas (predicates)
- State goal state as predicate formulas to achieve
- Use abstract logical reasoning - avoid physical interpretations
- Focus on predicate transformations rather than object manipulations

REQUIRED FORMAT: Provide a clear logical description of:
1. Available objects in logical terms
2. Initial predicate state (what logical formulas hold initially)
3. Goal predicate state (what logical formulas must hold finally)

Frame everything in terms of predicate logic operations and state transformations.
"""
        # Use concept modeler client for this task
        response = self.concept_modeler.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very deterministic for logical precision
            max_tokens=1000,
            timeout=TIMEOUT_SECONDS
        )

        return response.choices[0].message.content.strip()

    def solve_single_problem(self, domain: str, problem_id: str, problem_path: Path | None = None, domain_path: Path | None = None) -> dict:
        """Solve a single planning problem and return detailed results."""
        logger.info(f"Solving problem {problem_id} in domain {domain}")

        start_time = time.time()
        results = {
            "problem_id": problem_id,
            "domain": domain,
            "experiment_type": "mystery",
            "timestamp": time.time(),
            "success": False
        }

        try:
            # Step 1: Load domain and problem files
            domain_content, problem_content, domain_path = self.load_dataset_instance(
                domain, problem_id, problem_path_override=problem_path, domain_path_override=domain_path
            )
            problem_description = self.build_problem_description(problem_content)

            # Step 2: Concept modeling
            step_start = time.time()
            conceptual_model = self.get_concept_model(domain_path)
            results["concept_modeling_time"] = time.time() - step_start

            # Step 2.5: Parse PDDL into IR
            step_start = time.time()
            domain_ir = parse_domain(domain_content)
            problem_ir = parse_problem(problem_content, domain_ir)
            results["pddl_parsing_time"] = time.time() - step_start

            # Step 3-6: Plan → Semantics → Translate → VAL with iterative feedback (no hard-coding)
            attempts = []
            feedback = None
            max_retries = MAX_RETRIES
            pddl_actions = []
            validation_results = {"success": False, "status": "Not attempted"}
            for attempt in range(max_retries + 1):
                attempt_rec = {"attempt": attempt + 1}

                # Language planning (structured)
                step_start = time.time()
                natural_language_plan, planning_metadata, structured_plan = self.language_planner.create_plan(
                    domain_ir=domain_ir,
                    problem_ir=problem_ir,
                    problem_description=problem_description,
                    conceptual_model=conceptual_model,
                    problem_pddl=problem_content,
                    feedback=feedback,
                    domain_pddl=domain_content
                )
                results["language_planning_time"] = time.time() - step_start
                results["natural_language_plan"] = natural_language_plan
                results["planning_metadata"] = planning_metadata
                results["structured_plan"] = structured_plan
                attempt_rec["structured_plan"] = structured_plan

                # Pre-translation symbolic simulation (domain-agnostic STRIPS-like check)
                min_steps = 3 if ENFORCE_PLAN_LEN else 0
                if ENABLE_SIM:
                    ok, report = simulate_plan(domain_ir, problem_ir, structured_plan)
                else:
                    ok, report = True, "simulation skipped"
                attempt_rec["simulation"] = {"ok": ok, "report": report, "length": len(structured_plan)}
                if (not ok) or (len(structured_plan) < min_steps):
                    if len(structured_plan) < min_steps:
                        report = (report + "; " if report else "") + f"plan too short: {len(structured_plan)} steps (minimum {min_steps})"
                    feedback = (
                        "Simulation failed or insufficient plan length. Details: "
                        + (report or "no report")
                        + ". Please revise with 3–6 steps, ensuring each step's preconditions hold, updating the symbolic state with effects, "
                          "and achieving the goal in the final state. Use only allowed actions/objects and respect action arities."
                    )
                    attempts.append(attempt_rec)
                    continue  # retry planning

                # Semantic comprehension
                if self.semantics_comprehender:
                    step_start = time.time()
                    _ = self.semantics_comprehender.comprehend_semantics(
                        mystery_domain_content=domain_content,
                        natural_language_plan=natural_language_plan,
                        conceptual_model=conceptual_model
                    )
                    results["semantic_comprehension_time"] = time.time() - step_start

                # Translation with IR validation
                try:
                    step_start = time.time()
                    pddl_actions = self.pddl_translator.translate_to_pddl(
                        structured_plan, domain_ir, problem_ir, strict=TRANSLATE_STRICT
                    )
                    results["pddl_translation_time"] = time.time() - step_start
                    results["pddl_plan"] = pddl_actions
                    attempt_rec["pddl_plan"] = pddl_actions
                except Exception as e:
                    feedback = f"Plan validation failed before VAL: {str(e)}. " \
                               f"Please revise using only allowed actions/objects and produce a multi-step plan to achieve the goal."
                    attempt_rec["error"] = str(e)
                    attempts.append(attempt_rec)
                    continue  # retry planning

                # VAL validation
                step_start = time.time()
                validation_results = self.val_interface.validate_plan(domain_content, problem_content, pddl_actions)
                results["validation_time"] = time.time() - step_start
                results["val_results"] = validation_results
                attempt_rec["val_results"] = validation_results

                attempts.append(attempt_rec)

                if validation_results.get("success"):
                    break  # done
                else:
                    # Construct concise feedback for LLM to repair plan
                    stdout = validation_results.get("stdout", "")
                    stderr = validation_results.get("stderr", "")
                    fb_parts = ["The previous plan failed VAL validation.",
                                f"VAL status: {validation_results.get('status', 'unknown')}."]
                    if stdout:
                        fb_parts.append("VAL stdout (truncated): " + stdout[:500])
                    if stderr:
                        fb_parts.append("VAL stderr (truncated): " + stderr[:300])
                    feedback = "\n".join(fb_parts)

            results["attempts"] = attempts

            # Fallback: domain-agnostic STRIPS search (no hard-coding)
            if not validation_results.get("success"):
                search_plan = plan_via_search(domain_ir, problem_ir, max_depth=6, min_depth=3)
                results["search_plan"] = search_plan
                if search_plan:
                    try:
                        pddl_actions = self.pddl_translator.translate_to_pddl(
                            search_plan, domain_ir, problem_ir, strict=TRANSLATE_STRICT
                        )
                        results["pddl_plan_search"] = pddl_actions
                        validation_results = self.val_interface.validate_plan(
                            domain_content, problem_content, pddl_actions
                        )
                        results["val_results"] = validation_results
                    except Exception as e:
                        results["search_error"] = str(e)

            # Overall success
            results["success"] = validation_results["success"]
            results["total_execution_time"] = time.time() - start_time

            logger.info(f"Problem {problem_id} solved successfully: {results['success']}")

        except Exception as e:
            results["error"] = str(e)
            results["total_execution_time"] = time.time() - start_time
            logger.error(f"Failed to solve problem {problem_id}: {str(e)}")

        return results

    def run_batch_experiment(self, domain: str, instances_dir: Path, skip_ids: set = None, instances_subdir: str | None = None, skip_edges: bool = False) -> list:
        """Run batch experiments on all problems in a domain with per-instance persistence."""
        logger.info(f"Starting batch experiment on {domain} domain")

        results = []
        base = Path(PLANBENCH_INSTANCES_PATH) / domain

        # Select problem files according to instances_subdir if provided
        if instances_subdir:
            candidate_dir = base / instances_subdir
            problem_files = list(candidate_dir.glob("*.pddl")) if candidate_dir.exists() else []
        else:
            # Backward compatible: prefer mystery/generated_basic, else generated_basic
            mystery_basic_dir = base / "mystery" / "generated_basic"
            problem_files = list(mystery_basic_dir.glob("*.pddl")) if mystery_basic_dir.exists() else []
            if not problem_files:
                basic_pattern = base / "generated_basic"
                problem_files = list(basic_pattern.glob("*.pddl")) if basic_pattern.exists() else []

        if not problem_files:
            logger.error(f"No problem files found for domain {domain} (instances_subdir={instances_subdir})")
            return results

        # Sort numerically by instance id if possible
        def _numkey(p: Path) -> int:
            try:
                return int(p.stem.split("-")[-1])
            except Exception:
                return 1 << 30
        problem_files = sorted(problem_files, key=_numkey)

        # Optionally skip the first and last item (e.g., for 0..101 -> 1..100)
        if skip_edges and len(problem_files) >= 2:
            problem_files = problem_files[1:-1]

        logger.info(f"Selected {len(problem_files)} problem files from {instances_subdir or 'default'}")
        # Resume capability: skip problems already saved in instances_dir or provided resume sets
        todo_files = []
        skipped = 0
        sidset = set(skip_ids) if skip_ids else set()
        for pf in problem_files:
            pid = pf.stem
            if (instances_dir / f"{pid}.json").exists() or pid in sidset:
                skipped += 1
                continue
            todo_files.append(pf)
        if skipped:
            logger.info(f"Skipping {skipped} problems with existing results; {len(todo_files)} remaining")

        total = len(todo_files)
        completed = 0
        for problem_file in todo_files:
            problem_id = problem_file.stem
            logger.info(f"Processing problem: {problem_id}")
            result = self.solve_single_problem(
                domain, problem_id,
                problem_path=problem_file,
                domain_path=(base / "mystery" / "generated_domain.pddl") if (instances_subdir and "mystery" in instances_subdir) else (base / "generated_domain.pddl")
            )
            results.append(result)
            # Immediate per-instance save
            try:
                self.save_instance_result(instances_dir, problem_id, result)
            except Exception as e:
                logger.error(f"Failed to save instance {problem_id}: {e}")
            self.cleanup_legacy_result(domain, problem_id, OPENAI_MODEL, instances_dir)

            completed += 1
            logger.info(f"Completed {completed}/{total} problems")

        return results

    def run_batch_experiment_parallel(self, domain: str, workers: int, instances_dir: Path, skip_ids: set = None, instances_subdir: str | None = None, skip_edges: bool = False) -> list:
        """Run batch experiments in parallel on all problems in a domain with per-instance persistence."""
        logger.info(f"Starting parallel batch experiment on {domain} domain with {workers} workers")

        # Collect problem files according to selection logic
        results = []
        base = Path(PLANBENCH_INSTANCES_PATH) / domain

        if instances_subdir:
            candidate_dir = base / instances_subdir
            problem_files = list(candidate_dir.glob("*.pddl")) if candidate_dir.exists() else []
        else:
            mystery_basic_dir = base / "mystery" / "generated_basic"
            problem_files = list(mystery_basic_dir.glob("*.pddl")) if mystery_basic_dir.exists() else []
            if not problem_files:
                basic_pattern = base / "generated_basic"
                problem_files = list(basic_pattern.glob("*.pddl")) if basic_pattern.exists() else []

        if not problem_files:
            logger.error(f"No problem files found for domain {domain} (instances_subdir={instances_subdir})")
            return results

        # Sort numerically and optionally skip edges
        def _numkey(p: Path) -> int:
            try:
                return int(p.stem.split("-")[-1])
            except Exception:
                return 1 << 30
        problem_files = sorted(problem_files, key=_numkey)
        if skip_edges and len(problem_files) >= 2:
            problem_files = problem_files[1:-1]

        # Resume capability and remove hard cap: process all problems, skipping existing
        todo_files = []
        skipped = 0
        sidset = set(skip_ids) if skip_ids else set()
        for pf in problem_files:
            pid = pf.stem
            if (instances_dir / f"{pid}.json").exists() or pid in sidset:
                skipped += 1
                continue
            todo_files.append(pf)
        if skipped:
            logger.info(f"Skipping {skipped} problems with existing results; {len(todo_files)} remaining")
        else:
            logger.info(f"Found {len(todo_files)} problem files")

        # Define a per-task solver to avoid cross-thread shared state
        def solve_task(problem_file: Path) -> dict:
            pid = problem_file.stem
            # Create an isolated runner per thread
            runner = ExperimentRunner()
            res = runner.solve_single_problem(
                domain, pid,
                problem_path=problem_file,
                domain_path=(base / "mystery" / "generated_domain.pddl") if (instances_subdir and "mystery" in instances_subdir) else (base / "generated_domain.pddl")
            )
            # Persist immediately
            try:
                runner.save_instance_result(instances_dir, pid, res)
            except Exception as e:
                logger.error(f"Failed to save instance {pid}: {e}")
            runner.cleanup_legacy_result(domain, pid, OPENAI_MODEL, instances_dir)
            return res

        # Execute in parallel
        completed = 0
        total = len(todo_files)
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(solve_task, pf): pf for pf in todo_files}
            for fut in futures.as_completed(future_to_file):
                pf = future_to_file[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Parallel task failed for {pf.stem}: {e}")
                finally:
                    completed += 1
                    logger.info(f"Completed {completed}/{total} problems")

        return results

    def save_results(self, results: list, filename: str):
        """Save experiment results to file."""
        output_path = Path("results") / f"{filename}.{OUTPUT_FORMAT}"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="LLM Abstract Reasoning Validation System")
    parser.add_argument("--domain", choices=EXPERIMENT_DOMAINS, required=True,
                       help="Planning domain to use")
    parser.add_argument("--type", choices=["mystery"], default="mystery",
                       help="Experiment type (currently only mystery supported)")
    parser.add_argument("--mode", choices=["single", "batch"], required=True,
                       help="Experiment mode")
    parser.add_argument("--problem", help="Problem ID for single mode")
    parser.add_argument("--output", default="experiment_results",
                       help="Output filename prefix")
    parser.add_argument("--workers", type=int, default=int(os.getenv("PARALLEL_WORKERS", "4")),
                       help="Number of parallel workers for batch mode")
    parser.add_argument("--resume_dir", action="append", default=[],
                       help="Directories containing per-instance jsons to skip (can repeat)")
    parser.add_argument("--instances_subdir", default=None,
                       help="Instances subdirectory under dataset domain, e.g., generated_basic_3")
    parser.add_argument("--skip_edges", action="store_true",
                       help="Skip the first and last problem after sorting by instance id")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "single" and not args.problem:
        parser.error("--problem is required for single mode")

    # Initialize experiment runner
    runner = ExperimentRunner()

    # Prepare output base dir and per-instance directory (stable timestamp for this run)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name = OPENAI_MODEL
    run_dir_name = f"{timestamp}_{args.type}_{args.domain}_{args.mode}_{model_name}_{args.output}"
    base_dir = Path("results") / "runs" / run_dir_name
    instances_dir = base_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    if LEGACY_RESULTS_CLEANUP and not args.instances_subdir:
        legacy_dir = Path("results") / "mystery" / args.domain / "batch" / model_name
        if legacy_dir.exists():
            moved = 0
            for jf in legacy_dir.glob("*.json"):
                try:
                    target = instances_dir / jf.name
                    if not target.exists():
                        target.write_text(jf.read_text())
                    jf.unlink()
                    moved += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate legacy file {jf}: {e}")
            if moved:
                logger.info(f"Migrated {moved} legacy files from {legacy_dir} to {instances_dir}")

    if LEGACY_RESULTS_CLEANUP and not args.instances_subdir:
        runner.start_legacy_watcher(args.domain, model_name, instances_dir)

    # Build skip set from current run directory and optional resume dirs (legacy or previous runs)
    skip_ids = set(p.stem for p in instances_dir.glob("*.json"))
    if getattr(args, "resume_dir", None):
        for d in args.resume_dir:
            pdir = Path(d)
            if pdir.exists():
                for jf in pdir.glob("*.json"):
                    skip_ids.add(jf.stem)
    if skip_ids:
        logger.info(f"Loaded {len(skip_ids)} existing instance IDs to skip")

    if args.mode == "single":
        res = runner.solve_single_problem(args.domain, args.problem)
        results = [res]
        # Persist the single instance immediately
        try:
            runner.save_instance_result(instances_dir, args.problem, res)
        except Exception as e:
            logger.error(f"Failed to save instance {args.problem}: {e}")
        runner.cleanup_legacy_result(args.domain, args.problem, model_name, instances_dir)
    else:
        if args.workers and args.workers > 1:
            results = runner.run_batch_experiment_parallel(
                args.domain, args.workers, instances_dir, skip_ids, args.instances_subdir, args.skip_edges
            )
        else:
            results = runner.run_batch_experiment(
                args.domain, instances_dir, skip_ids, args.instances_subdir, args.skip_edges
            )

    # Save results (optimized: results/runs/{run_dir_name}/summary.json)
    subpath = f"runs/{run_dir_name}/summary"
    runner.save_results(results, subpath)

    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"\nExperiment Summary:")
    print(f"Domain: {args.domain}")
    print(f"Mode: {args.mode}")
    print(f"Type: {args.type}")
    print(f"Problems processed: {total}")
    print(f"Successful solutions: {successful}")
    print(f"Success rate: {successful/total:.2%}" if total > 0 else "Success rate: N/A")

if __name__ == "__main__":
    main()
