from rich.logging import RichHandler
import logging

log = logging.getLogger("rec_engine")
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    handler = RichHandler(show_path=False)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    log.addHandler(handler)
