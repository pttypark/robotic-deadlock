from deadlock.dsd_fsd.model import SystemType

__all__ = ["SystemType", "run_once", "run_comparison"]


def __getattr__(name):
    if name in {"run_once", "run_comparison"}:
        from deadlock.dsd_fsd.experiment import run_comparison, run_once

        return {"run_once": run_once, "run_comparison": run_comparison}[name]
    raise AttributeError(name)
