import asyncio
import os
from pathlib import Path
import warnings

import nbformat
from nbclient import NotebookClient
import jupyter_core.paths

warnings.filterwarnings(
    "ignore",
    message=".*WindowsSelectorEventLoopPolicy.*",
    category=DeprecationWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "docs" / "image_dehazing_aodnet_residual_skip_notebook.ipynb"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "image_dehazing_aodnet_residual_skip_notebook.executed.ipynb"


def prepare_workspace_dirs():
    # Keep every runtime/config/cache folder inside the project directory.
    for dirname in [
        ".jupyter_runtime",
        ".jupyter_config",
        ".jupyter_data",
        ".ipython",
        ".mplconfig",
    ]:
        (PROJECT_ROOT / dirname).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(PROJECT_ROOT / ".jupyter_runtime"))
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(PROJECT_ROOT / ".jupyter_config"))
    os.environ.setdefault("JUPYTER_DATA_DIR", str(PROJECT_ROOT / ".jupyter_data"))
    os.environ.setdefault("IPYTHONDIR", str(PROJECT_ROOT / ".ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


def disable_windows_acl_guard():
    # In restricted Windows environments Jupyter's secure ACL call can fail
    # even for writable project-local folders. A no-op keeps execution local.
    jupyter_core.paths.win32_restrict_file_to_user = lambda *_args, **_kwargs: None


def main():
    # This avoids zmq issues that sometimes appear on Windows with notebook execution.
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    prepare_workspace_dirs()
    disable_windows_acl_guard()

    # Read the notebook, execute every cell, then save the executed copy.
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)

    notebook = nbformat.validator.normalize(notebook)[1]

    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)

    print(f"Executed notebook written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
