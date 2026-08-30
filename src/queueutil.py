"""Shared best-effort queue delivery helper.

The worker modules (audio, transcriber, translator) each deliver events to
queues without ever letting a failed put kill the pipeline. This module
centralizes that logic; each module keeps its own thin wrapper so callers
and tests that patch those wrappers are unaffected.
"""

import logging
import queue

logger = logging.getLogger(__name__)


def put_best_effort(q, msg, block=False, timeout=None, error_msg=None, debug_msg=None):
    """Put ``msg`` on queue ``q`` without ever raising.

    Returns True on success, False on failure. Logging policy:
      * ``queue.Full`` while blocking -> ERROR with ``error_msg`` (if given);
      * ``queue.Full`` when no ``error_msg`` was provided (caller relied on
        debug logging, e.g. ``_put_ui``) -> DEBUG with ``debug_msg``;
      * any other failure -> DEBUG with ``debug_msg`` (if given).
    """
    try:
        q.put(msg, block=block, timeout=timeout)
        return True
    except queue.Full as e:
        if block and error_msg:
            logger.error(error_msg)
        elif error_msg is None and debug_msg:
            logger.debug(f"{debug_msg}: {e}")
        return False
    except Exception as e:
        if debug_msg:
            logger.debug(f"{debug_msg}: {e}")
        return False
