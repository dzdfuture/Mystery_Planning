# Mystery Planning

An LLM-based abstract reasoning validation system for evaluating large language models on PDDL planning tasks. The system implements a complete planning pipeline including problem understanding, plan generation, validation, and analysis, specifically designed for mysterious blocksworld domains.

## Data Sources

The content in the `data/` folder is sourced from the [PlanBench paper](https://arxiv.org/abs/2210.07128), including:
- LLM planning analysis tools (`llm_planning_analysis`)
- PDDL generators and executors
- Instance data for multiple planning domains
- VAL plan validation tools

Other folders contain our implementations:
- `src/`: Core implementation code
- `results/`: Experimental results and analyses
- `agent_workspace/`: Agent workspace

## Project Structure

### Core Modules (`src/`)

- **main.py**: Main entry point for running experiments, implementing single problem solving and batch experiments
- **concept_modeler.py**: Concept modeling module that builds domain knowledge models from PDDL domain files
- **language_planner.py**: Language planner that uses LLMs to generate natural language planning descriptions
- **semantics_comprehender.py**: Semantics comprehension module for domain-specific understanding
- **pddl_translator.py**: PDDL translator that converts structured plans to executable PDDL action sequences
- **val_interface.py**: VAL validator interface for formal plan validation
- **pddl_parser.py**: PDDL parser for parsing domain and problem files
- **strips_search.py**: STRIPS search implementation providing search baseline
- **strips_sim.py**: STRIPS simulator for symbolic-level state simulation

## Installation

### System Dependencies

This project requires the following system tools:

- **g++ / GCC**: For compiling C++ code
- **make**: Build tool
- **flex**: Lexical analyzer generator
- **bison**: Parser generator

Install on Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential flex bison
```

The VAL validator comes pre-compiled. To recompile if needed:
```bash
cd data/planbench/planner_tools/VAL
make validate  # Compile validator
make parser    # Compile parser
```

### Python Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Mystery_Planning
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Single Problem Solving

Run a single mysterious blocksworld problem:

```bash
python src/main.py --domain mystery_blocksworld --mode single --problem instance-1
```

### Batch Experiments

Run batch experiments with 4 parallel workers:

```bash
python src/main.py --domain mystery_blocksworld --mode batch --workers 4
```

### Advanced Options

- `--instances_subdir`: Specify instance subdirectory
- `--skip_edges`: Skip edge instances
- `--resume_dir`: Resume previous experiments

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project uses the appropriate license. Please refer to the LICENSE file for details.
