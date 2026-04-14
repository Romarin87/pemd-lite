from .project import Project, load
from .table import generate_projects_from_table

__version__ = "0.1.0"

__all__ = ["Pipeline", "Project", "generate_projects_from_table", "load", "run_polymer_pack_md", "run_until"]


def __getattr__(name):
    if name in {"Pipeline", "run_polymer_pack_md", "run_until"}:
        from .pipeline import Pipeline, run_polymer_pack_md, run_until

        mapping = {
            "Pipeline": Pipeline,
            "run_polymer_pack_md": run_polymer_pack_md,
            "run_until": run_until,
        }
        return mapping[name]
    raise AttributeError(name)
