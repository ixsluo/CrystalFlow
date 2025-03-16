import os
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ["PACKAGE_ROOT"] = str(PACKAGE_ROOT)


# Set `eval` resolver
def try_eval(s):
    """This is a custom resolver for OmegaConf that allows us to use `eval` in our config files
    with the syntax `${eval:'${foo} + ${bar}'}

    See:
    https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
    """
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise


def cond_parser(value):
    condition, true_val, false_val = value.split(',')
    return true_val if eval(condition) else false_val


OmegaConf.register_new_resolver("eval", try_eval)
OmegaConf.register_new_resolver("cond", cond_parser)