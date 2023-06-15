from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

# this ensures all logging levels get marked with the rank zero decorator
# otherwise logs would get multiplied for each GPU process in multi-GPU setup
logging_levels = (
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
)
for level in logging_levels:
    setattr(logger, level, rank_zero_only(getattr(logger, level)))
