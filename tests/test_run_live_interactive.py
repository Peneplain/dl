"""Exercise the interactive viewer startup without loading model weights."""

from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_live import run_interactive


class InteractiveViewerStartupTests(unittest.TestCase):
    def test_idle_viewer_is_created_and_refreshed_before_first_prompt(self):
        order = []
        viewer_created = threading.Event()
        idle_refreshed = threading.Event()

        class InputAfterViewer:
            def __iter__(self):
                viewer_created.wait(timeout=3)
                idle_refreshed.wait(timeout=3)
                yield "quit\n"

        class Service:
            def __init__(self, args):
                order.append("ardy")

        class Policy:
            def __init__(self, *args, **kwargs):
                order.append("sonic")

        class Simulation:
            def __init__(self, *args, **kwargs):
                order.append("viewer")
                viewer_created.set()

            def sync_viewer(self):
                idle_refreshed.set()

            def close(self):
                order.append("closed")

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                out=Path(directory) / "session", video=None, assets=Path("assets"),
                sonic_repo=Path("sonic"), gui=True, prompt=None,
            )
            with patch("scripts.run_live.ArdyService", Service), \
                    patch("scripts.run_live.SonicPolicy", Policy), \
                    patch("scripts.run_live.SonicSimulation", Simulation), \
                    patch("scripts.run_live.sys.stdin", InputAfterViewer()):
                run_interactive(args)

        self.assertEqual(order, ["ardy", "sonic", "viewer", "closed"])
        self.assertTrue(idle_refreshed.is_set())


if __name__ == "__main__":
    unittest.main()
