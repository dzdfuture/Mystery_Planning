# Mystery Planning

A symbol-agnostic closed-loop planning pipeline for evaluating large language models' structural reasoning capabilities under lexical-prior-free conditions. The system implements a complete "generate→verify→repair" cycle for PDDL planning tasks in Mystery domains, where all predicates and action names are replaced with semantically irrelevant random symbols while preserving logical structures. This design eliminates vocabulary-based priors, enabling pure evaluation of models' structural reasoning abilities.

## Data Sources

The content in the `data/` folder is sourced from the [PlanBench paper](https://neurips.cc/virtual/2023/poster/73553), including:
- LLM planning analysis tools (`llm_planning_analysis`)
- PDDL generators and executors
- Instance data for multiple planning domains
- VAL plan validation tools

Other folders contain our implementations:
- `src/`: Core implementation code
- `results/`: Experimental results and analyses
- `agent_workspace/`: Agent workspace

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
