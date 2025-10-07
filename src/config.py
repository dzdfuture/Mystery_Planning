# Configuration for the LLM Abstract Reasoning Validation System
# - No hard-coded secrets or absolute paths
# - All configurable fields are read from environment variables with safe fallbacks
# - Paths default to repository-relative locations for portability

import os
from pathlib import Path

# Project root (repo root assumed two levels above this file: src/...)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required at runtime
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o1-mini")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Experiment Configuration
EXPERIMENT_DOMAINS = ["blocksworld", "logistics"]
EXPERIMENT_TYPES = ["mystery"]

# PlanBench Dataset Configuration
# Allow override via env, else default to repo-local dataset
PLANBENCH_DATA_PATH = os.getenv(
    "PLANBENCH_DATA_PATH",
    str(PROJECT_ROOT / "data" / "planbench" / "plan-bench")
)
PLANBENCH_INSTANCES_PATH = os.path.join(PLANBENCH_DATA_PATH, "instances")
PLANBENCH_RESULTS_PATH = os.path.join(PLANBENCH_DATA_PATH, "results")

# VAL Validator Configuration
# Allow override via env, else default to repo-local VAL build bundled under data/planbench/planner_tools/VAL
VAL_ROOT = os.getenv(
    "VAL_ROOT",
    str(PROJECT_ROOT / "data" / "planbench" / "planner_tools" / "VAL")
)
VAL_EXECUTABLE_PATH = os.getenv("VAL_EXECUTABLE_PATH", str(Path(VAL_ROOT) / "validate"))
VAL_PARSER_EXECUTABLE = os.getenv("VAL_PARSER_EXECUTABLE", str(Path(VAL_ROOT) / "parser"))

# Output Configuration
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Batch Processing Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))  # seconds
# Retry & feature flags (tunable to control API usage and pipeline stages)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # default fewer retries to reduce API cost
ENABLE_SEMANTICS = os.getenv("ENABLE_SEMANTICS", "0") == "1"  # disable semantics stage by default
ENABLE_CONCEPT = os.getenv("ENABLE_CONCEPT", "1") == "1"      # enable conceptual modeling by default
ENABLE_PROBLEM_DESC_LLM = os.getenv("ENABLE_PROBLEM_DESC_LLM", "0") == "1"  # default: local description
LEGACY_RESULTS_CLEANUP = os.getenv("LEGACY_RESULTS_CLEANUP", "1") == "1"    # clean stray legacy files
# Simulation switch for ablation (S-1 No-Sim): 1=enable STRIPS simulation, 0=skip simulation but keep VAL
ENABLE_SIM = os.getenv("ENABLE_SIM", "1") == "1"
ENFORCE_PLAN_LEN = os.getenv("ENFORCE_PLAN_LEN", "1") == "1"
# Translation strictness for ablation: 1=strict translation (default), 0=relaxed
TRANSLATE_STRICT = os.getenv("TRANSLATE_STRICT", "1") == "1"
