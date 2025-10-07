# VAL Interface - Integration with VAL validator for PDDL plan verification

import subprocess
import os
import tempfile
import logging
from src.config import VAL_EXECUTABLE_PATH, VAL_PARSER_EXECUTABLE
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VALInterface:
    def __init__(self):
        self.validate_executable = VAL_EXECUTABLE_PATH
        self.parser_executable = VAL_PARSER_EXECUTABLE

        # Verify VAL executables exist
        if not os.path.exists(self.validate_executable):
            raise FileNotFoundError(f"VAL validate executable not found at: {self.validate_executable}")
        if not os.path.exists(self.parser_executable):
            raise FileNotFoundError(f"VAL parser executable not found at: {self.parser_executable}")

        # Make executables executable if needed
        for exe in [self.validate_executable, self.parser_executable]:
            if not os.access(exe, os.X_OK):
                os.chmod(exe, 0o755)  # Add execute permission

    def validate_plan(self, domain_pddl: str, problem_pddl: str, plan_actions: list) -> Dict[str, Any]:
        """
        Validate a PDDL plan using VAL and return detailed validation results.

        Args:
            domain_pddl (str): Domain PDDL content
            problem_pddl (str): Problem PDDL content
            plan_actions (list): List of action strings in PDDL format

        Returns:
            Dict[str, Any]: Validation results including success status and details
        """
        try:
            # Create temporary files for VAL input
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as domain_file:
                domain_file.write(domain_pddl)
                domain_file_path = domain_file.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as problem_file:
                problem_file.write(problem_pddl)
                problem_file_path = problem_file.name

            # Convert action format for VAL compatibility
            formatted_actions = []
            for action in plan_actions:
                if isinstance(action, str):
                    # Convert "action(param1,param2)" to "(action param1 param2)"
                    if '(' in action and ')' in action:
                        action_name = action.split('(')[0].strip()
                        params = action.replace(action_name, '').strip().strip('()')
                        # Replace commas with spaces to match PDDL format
                        clean_params = params.replace(',', ' ')
                        if clean_params.strip():
                            formatted_action = f"({action_name} {clean_params})"
                        else:
                            formatted_action = f"({action_name})"
                        formatted_actions.append(formatted_action)
                    else:
                        formatted_actions.append(action)

            # Create plan file with proper PDDL format
            plan_content = '\n'.join(formatted_actions) + '\n'
            with tempfile.NamedTemporaryFile(mode='w', suffix='.plan', delete=False) as plan_file:
                plan_file.write(plan_content)
                plan_file_path = plan_file.name
                
            logger.info(f"VAL Plan Format: {plan_content.strip()}")

            try:
                # Run VAL validation
                cmd = [
                    self.validate_executable,
                    '-v',
                    domain_file_path,
                    problem_file_path,
                    plan_file_path
                ]

                logger.info(f"Running VAL validation: {' '.join(cmd)}")

                # Execute VAL and capture output
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout for validation
                )

                logger.info(f"VAL return code: {result.returncode}")
                logger.debug(f"VAL stdout: {result.stdout}")
                if result.stderr:
                    logger.warning(f"VAL stderr: {result.stderr}")

                # Parse validation results
                validation_results = self._parse_val_output(result.returncode, result.stdout, result.stderr)

                return validation_results

            finally:
                # Clean up temporary files
                for temp_file in [domain_file_path, problem_file_path, plan_file_path]:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        logger.warning(f"Could not delete temporary file: {temp_file}")

        except subprocess.TimeoutExpired:
            logger.error("VAL validation timed out")
            return {
                "success": False,
                "error": "Validation timed out",
                "execution_time": None
            }
        except Exception as e:
            logger.error(f"Error during VAL validation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": None
            }

    def _parse_val_output(self, return_code: int, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parse VAL output to extract validation results.

        Args:
            return_code (int): VAL process return code
            stdout (str): Standard output from VAL
            stderr (str): Standard error from VAL

        Returns:
            Dict[str, Any]: Parsed validation results
        """
        validation_result = {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": None
        }

        # Extract execution time if present
        import re
        time_match = re.search(r'(\d+\.\d+) seconds', stdout)
        if time_match:
            validation_result["execution_time"] = float(time_match.group(1))

        # Extract key information from output
        if "Successful plans:" in stdout:
            validation_result["status"] = "Plan validated successfully"
        elif "Failed plans:" in stdout:
            validation_result["status"] = "Plan validation failed"
        elif "Cannot instantiate ground action" in stdout:
            validation_result["status"] = "Plan contains invalid actions"
        else:
            validation_result["status"] = "Unknown validation status"

        return validation_result
