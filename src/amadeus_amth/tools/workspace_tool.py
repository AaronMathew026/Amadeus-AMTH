"""Workspace tools — let the model read, write and organise files on disk."""

import functools
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Anchored to this file, not the cwd, so the paths are the same no matter where
# the process was launched from — same approach as memory/semantic_memory.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Loaded here rather than relying on the caller: engine/client.py imports this
# package before it calls load_dotenv(), so reading the variable at import time
# would otherwise always miss it. load_dotenv() is idempotent.
load_dotenv()

# The agent's own scratch space — notes, logs, anything it writes for itself.
# Override with AMADEUS_WORKSPACE in .env; defaults to <project>/workspace.
WORKSPACE_ROOT = Path(
    os.getenv("AMADEUS_WORKSPACE") or PROJECT_ROOT / "workspace"
).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


class WorkspaceError(Exception):
    """Raised when a tool is asked for a path outside the workspace. Reported to
    the model as an ordinary "Error: ..." string, like every other refusal."""


def _tool(fn):
    """Turn the exceptions a filesystem call can raise into the error strings
    these tools already return, so a refused path, a permission problem or a
    binary file never unwinds into the tool-call loop."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except WorkspaceError as e:
            return f"Error: {e}"
        except UnicodeDecodeError:
            return f"Error: {fn.__name__} failed: not UTF-8 text (binary file?)."
        except OSError as e:
            return f"Error: {fn.__name__} failed: {e}"

    return wrapper


def _exists(p: Path) -> bool:
    """Path.exists() follows symlinks, so it reports False for a broken one. A
    broken link is still something the agent can see and delete."""
    return p.exists() or p.is_symlink()


def _safe_path(path: str, *, follow: bool = True) -> Path:
    """Make `path` absolute (relative paths hang off WORKSPACE_ROOT) and confine
    it to the workspace.

    Parent directories are resolved before the containment check, so neither a
    '..' nor a symlinked directory part-way along the path can walk out. The
    final component is left unresolved on purpose: delete and move need to see a
    symlink itself rather than whatever it points at.

    follow=False skips the check on a final symlink's target — only for callers
    that act on the link and never on the target.
    """
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p

    # '..' and '' as the final component have no name worth preserving, and
    # would slip through the check below because is_relative_to() is lexical:
    # '<root>/..' still looks like it sits under '<root>'. Resolve those fully.
    full = p.resolve() if p.name in ("", "..") else p.parent.resolve() / p.name

    if not full.is_relative_to(WORKSPACE_ROOT):
        raise WorkspaceError(f"'{path}' is outside the workspace ({WORKSPACE_ROOT}).")
    if follow and full.is_symlink():
        if not full.resolve().is_relative_to(WORKSPACE_ROOT):
            raise WorkspaceError(
                f"'{path}' is a symlink pointing outside the workspace."
            )
    return full


@_tool
def read_file(path: str) -> str:
    """Read a file from the workspace and return its contents as a string."""
    full = _safe_path(path)
    if not full.exists():
        return f"Error: '{full}' does not exist."
    if full.is_dir():
        return f"Error: '{full}' is a directory, not a file."
    return full.read_text(encoding="utf-8")


@_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace, creating directories as needed."""
    full = _safe_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to '{path}'."


@_tool
def list_files(path: str = ".") -> str:
    """List all files in the given directory, traversing subdirectories recursively,
    and return them as POSIX-style relative paths. Symlinked directories are not
    descended into."""
    root = _safe_path(path)
    if not root.exists() or not root.is_dir():
        return f"Error: '{path}' is not a valid directory."
    matches = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return "\n".join(matches) if matches else "No files found."


@_tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string."""
    full = _safe_path(path)
    if not full.exists():
        return f"Error: '{full}' does not exist."
    content = full.read_text(encoding="utf-8")
    if old_string not in content:
        return f"Error: The string to replace was not found in '{full}'."

    full.write_text(content.replace(old_string, new_string), encoding="utf-8")
    return f"Successfully replaced occurrences in '{full}'."


@_tool
def delete_file(path: str) -> str:
    """Delete a file, directory or symlink. A symlink is removed itself — the
    tree it points at is left alone."""
    full = _safe_path(path, follow=False)
    if not _exists(full):
        return f"Error: '{full}' does not exist."
    # unlink() on a symlink removes the link whether it points at a file or a
    # directory; only a real directory should ever reach rmtree.
    if full.is_symlink() or full.is_file():
        full.unlink()
        return f"Successfully deleted '{full}'."
    shutil.rmtree(full)
    return f"Successfully deleted directory '{full}'."


@_tool
def move_file(source_path: str, destination_path: str) -> str:
    """Move or rename a file, directory or symlink."""
    src = _safe_path(source_path, follow=False)
    dst = _safe_path(destination_path, follow=False)
    if not _exists(src):
        return f"Error: Source '{src}' does not exist."
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Successfully moved '{src}' to '{dst}'."


@_tool
def read_directory(path: str = ".") -> str:
    """List files and directories in the given directory (non-recursive)."""
    root = _safe_path(path)
    if not root.exists() or not root.is_dir():
        return f"Error: '{path}' is not a valid directory."
    items = []
    for p in root.iterdir():
        # Tested first: a link to a directory is also is_dir().
        kind = "LINK" if p.is_symlink() else "DIR " if p.is_dir() else "FILE"
        items.append(f"[{kind}] {p.name}")
    return "\n".join(items) if items else "Directory is empty."


@_tool
def file_exists(path: str) -> str:
    """Check if a file or directory exists."""
    return "True" if _exists(_safe_path(path, follow=False)) else "False"


@_tool
def search_files(query: str, path: str = ".") -> str:
    """Search for files matching a glob pattern. Symlinked directories are not
    descended into. Matching is case-sensitive on Linux and macOS."""
    root = _safe_path(path)
    if not root.exists() or not root.is_dir():
        return f"Error: '{path}' is not a valid directory."
    matches = [p.relative_to(root).as_posix() for p in root.rglob(query)]
    return "\n".join(matches) if matches else f"No files matched '{query}'."


@_tool
def create_directory(path: str) -> str:
    """Create a new directory."""
    full = _safe_path(path)
    if _exists(full):
        return f"Error: '{full}' already exists."
    full.mkdir(parents=True)
    return f"Successfully created directory '{full}'."


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a file and returns it as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file, creating any missing directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Lists all files in a directory, traversing subdirectories "
                "recursively, and returns them as relative paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the directory. Relative paths resolve from the workspace root; paths outside the workspace are refused. "
                            "Omit for the workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edits a file by replacing an exact string with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "The exact string to find and replace. Must match "
                            "exactly, including whitespace."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace the old string with.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Deletes a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file or directory to delete. Relative paths resolve from the workspace root; paths outside the workspace are refused."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Moves or renames a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": (
                            "Path to the source file or directory. Relative paths resolve from the workspace root; paths outside the workspace are refused."
                        ),
                    },
                    "destination_path": {
                        "type": "string",
                        "description": "Path to the destination. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    },
                },
                "required": ["source_path", "destination_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_directory",
            "description": "Lists files and directories in the given directory (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the directory. Relative paths resolve from the workspace root; paths outside the workspace are refused. "
                            "Omit for the workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_exists",
            "description": "Checks if a file or directory exists at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Searches for files matching a glob pattern recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The glob pattern to search for (e.g., '*.py', 'test_*')."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the directory to search in. Relative paths resolve from the workspace root; paths outside the workspace are refused. "
                            "Omit for the workspace root."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": (
                "Creates a new directory, including any necessary parent directories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to create. Relative paths resolve from the workspace root; paths outside the workspace are refused.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "move_file": move_file,
    "read_directory": read_directory,
    "file_exists": file_exists,
    "search_files": search_files,
    "create_directory": create_directory,
}
