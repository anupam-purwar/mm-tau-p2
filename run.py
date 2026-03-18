import os
import sys
import logging
import argparse
import datetime

from config import *

from simulator.Tau2BenchSimulator import Tau2VoiceSimulator
from simulator.MMTauEval import MultiModalTauEval
from tau2.run import get_tasks

logger = logging.getLogger(__name__)

# writes output to a log file, duplicates to the terminal if instructed
class StreamWriter:

    def __init__(self, std, log_file, duplicate = True):
        self.std = std
        self.log_file = log_file
        self.duplicate = duplicate

    def write(self, data):
        self.log_file.write(data)
        self.log_file.flush()
        if self.duplicate: # write to stderr/stdout if duplicate set to True
            self.std.write(data)
            self.std.flush()

    def flush(self):
        self.std.flush()
        self.log_file.flush()

# Context manager that redirects stdout and stderr to StreamWriter
class OutputLogger:

    def __init__(self, log_file_path: str, duplicate: bool = False):
        self.log_file_path = log_file_path
        self.duplicate = duplicate

    def __enter__(self):

        os.makedirs(os.path.dirname(os.path.abspath(self.log_file_path)), exist_ok=True)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._log_file = open(self.log_file_path, "w")

        sys.stdout = StreamWriter(self._orig_stdout, self._log_file)
        sys.stderr = StreamWriter(self._orig_stderr, self._log_file)

        self._log_handler = logging.FileHandler(self.log_file_path)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(self._log_handler)

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler.close()

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._log_file.close()
        return False

def run_pipeline(
    domain,
    eval_path='',
    start_idx=0,
    stop_idx=None,
    split='test',
    inject_persona=False,
    inject_context=False,
    metrics_to_skip=None,
    run_name='MMTauPipeline',
    duplicate=True,
    save=True,
):
    """End-to-end MMTau pipeline: simulate voice conversations then evaluate

    Args:
        domain: Task domain ('retail' or 'telecom')
        eval_path: If path is provided, skip simulation and only run evaluation on existing data
        start_idx: First task index to process (inclusive)
        stop_idx: Last task index to process (exclusive); None runs all
        split: Dataset split name (e.g. 'test', 'train')
        inject_persona: Augment agent prompt with user persona from task details
        inject_context: Augment agent prompt with inferred conversational context
        metrics_to_skip: List of metric names to exclude from evaluation
        run_name: Prefix for the run directory and file naming
        duplicate: Mirror log output to the terminal
        save: Persist intermediate simulation files

    Returns:
        Tuple of (metrics dict, run directory path).
    """
    run_dir = f"{TEMP_DIR}/{run_name+str(datetime.datetime.now()).replace(' ','-')}"
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "pipeline.log")

    with OutputLogger(log_path, duplicate=duplicate):
        logger.info("Pipeline started. Run directory: %s", run_dir)

        if not eval_path: # run simulation if eval_path not provided
            tasks = get_tasks(domain, task_split_name=split)
            stop_idx = min(len(tasks), stop_idx) if stop_idx is not None else len(tasks)

            for idx in range(start_idx, stop_idx):
                task = tasks[idx]
                logger.info("RUNNING TASK %s %d %s", split, idx, task.id)

                sim = Tau2VoiceSimulator(
                    task,
                    domain=domain,
                    inject_persona=inject_persona,
                    inject_context=inject_context,
                    username=run_name,
                    temp_dir=run_dir,
                )
                
                sim.converse_voice(save=True)
                del sim
        
        else: # if eval_path provided, point run_dir to it
            run_dir = eval_path

        critical_fields = CRITICAL_FIELDS
        if metrics_to_skip is None:
            metrics_to_skip = []

        evaluator = MultiModalTauEval(
            critical_fields=critical_fields,
            metrics_to_skip=metrics_to_skip,
            model=EVAL_MODEL,
        )

        metrics = evaluator.eval(dir_path=run_dir)

    if not save: # delete intermediate simulation files if save set to False
        for file in os.listdir(run_dir):
            if file.startswith(run_name):
                os.remove(file)

    logger.info("Pipeline complete. Results saved to: %s", run_dir)
    print(metrics)
    return metrics, run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMTau-p2 Pipeline",
    )
    parser.add_argument("--domain", choices=["retail", "telecom"], default=DOMAIN, help="Task domain")
    parser.add_argument("--eval_path", type=str, default='', help="Run only evaluations on existing run (default: False)")
    parser.add_argument("--run_name", default="MMTauPipeline", help="Run name prefix (default: MMTauPipeline)")
    parser.add_argument("--start", type=int, default=0, help="Start task index (default: 0)")
    parser.add_argument("--stop", type=int, default=None, help="Stop task index exclusive (default behaviour: run all)")
    parser.add_argument("--split", default="test", help="Task split name (default: test)")
    parser.add_argument("--inject-persona", default=INJECT_PERSONA, action="store_true", help="Inject user persona into agent prompt")
    parser.add_argument("--inject-context", default=INJECT_CONTEXT,action="store_true", help="Inject user context into agent prompt")
    parser.add_argument("--critical-fields", nargs="*", default=None, help="Critical fields for evaluation (default: domain-specific)")
    parser.add_argument("--metrics-to-skip", nargs="*", default=None, help="Metric names to exclude from evaluation")
    parser.add_argument("--duplicate", action="store_true", help="Write terminal output alongside log file", default=True)
    parser.add_argument("--save", action="store_true", help="Save the intermediate files generated", default=True)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    metrics, run_dir = run_pipeline(
        domain=args.domain,
        eval_path=args.eval_path,
        start_idx=args.start,
        stop_idx=args.stop,
        split=args.split,
        inject_persona=args.inject_persona,
        inject_context=args.inject_context,
        metrics_to_skip=args.metrics_to_skip,
        run_name=args.run_name,
        duplicate=args.duplicate,
        save=args.save
    )

    print(f"\nRun directory: {run_dir}")
    print(f"Metrics: {metrics}")
