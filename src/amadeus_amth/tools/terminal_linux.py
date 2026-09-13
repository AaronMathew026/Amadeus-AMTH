"""Terminal tools — let the model open a real shell on the Ubuntu host.

Two ways in:

  * run_command      one-shot. Run a line, wait for it, get the output back.
  * terminal_*       a persistent bash session on a pty, which keeps its
                     working directory, its environment and whatever program is
                     still running in it between calls — a terminal the agent
                     opens once and keeps using.

Every command taken by either route goes through check_command() first, and
what that lets through is set by the ACCESS_LEVEL constant below.

What the levels are and are not. They are pattern checks on a command line,
meant to stop the model doing something careless — deleting a system directory
because it misread a path, rebooting the box to "fix" a service. Pattern checks
on a shell command can always be worked around by anything actively trying to:
a variable, base64, a script file, an interpreter. So this is a guard rail, not
a sandbox, and not a security boundary. The boundary has to come from the OS:
run the agent as an unprivileged user with no sudo rights, ideally in a
container, with nothing mounted that it has no business reaching. ACCESS_LEVEL
is the seatbelt, not the roll cage.
"""

import atexit
import enum
import functools
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .workspace_tool import PROJECT_ROOT, WORKSPACE_ROOT


class AccessLevel(enum.IntEnum):
    """How much of the host the model is allowed to touch.

    RESTRICTED    An allowlist of read-only commands, run without a shell, with
                  the working directory confined to the workspace. No pipes, no
                  redirection, no chaining — look, don't touch.
    MEDIUM        A full shell and the run of the filesystem, minus anything
                  that administers the box: no sudo, no package installs, no
                  service/user/firewall/mount changes, no writing into system
                  directories, no containers, no `curl | sh`.
    UNRESTRICTED  Everything except the handful of commands in _CRITICAL_RULES,
                  which are refused at every level because there is no coming
                  back from them (wiping /, formatting a disk, powering off).
    """

    RESTRICTED = 0
    MEDIUM = 1
    UNRESTRICTED = 2


# ---------------------------------------------------------------------------
# The dial. Change this line to change what the agent may run.
# AMADEUS_TERMINAL_ACCESS in .env ("restricted", "medium", "unrestricted")
# overrides it at deploy time, so the container can be handed a wider level
# than a laptop checkout without editing the source.
# ---------------------------------------------------------------------------
ACCESS_LEVEL = AccessLevel.UNRESTRICTED

_OVERRIDE = (os.getenv("AMADEUS_TERMINAL_ACCESS") or "").strip().upper()
if _OVERRIDE in AccessLevel.__members__:
    ACCESS_LEVEL = AccessLevel[_OVERRIDE]


# How long a command may run before it is killed, and how long it may run if
# the model asks for longer. Work that needs more than this belongs in a
# session, where a slow command keeps running after terminal_run() returns.
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = {
    AccessLevel.RESTRICTED: 60,
    AccessLevel.MEDIUM: 300,
    AccessLevel.UNRESTRICTED: 900,
}

# Output goes straight back into the context window, so a command that prints a
# megabyte gets its middle cut out rather than the conversation.
MAX_OUTPUT_CHARS = 20_000

# Sessions are cheap but they are real processes; this stops a confused model
# from opening one per turn forever.
MAX_SESSIONS = 4

_POSIX = os.name == "posix"
SHELL = "/bin/bash" if _POSIX else (shutil.which("bash") or "")

try:  # pty/termios are POSIX-only — the module still has to import on Windows.
    import fcntl
    import pty
    import termios

    _PTY_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows dev checkouts
    _PTY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Commands RESTRICTED may run. Every one of them reads something and writes
# nothing. Deliberately absent: sed, awk, perl, python and the other
# interpreters, because each is a way to write files and spawn processes while
# looking like a text filter.
RESTRICTED_COMMANDS = frozenset(
    {
        # files and directories
        "ls", "cat", "head", "tail", "wc", "stat", "file", "find", "tree",
        "basename", "dirname", "readlink", "realpath", "diff", "cmp",
        "md5sum", "sha256sum", "du", "df", "pwd",
        # text
        "grep", "egrep", "fgrep", "rg", "sort", "uniq", "cut", "tr", "echo",
        "printf",
        # host and processes
        "uname", "hostname", "whoami", "id", "groups", "date", "uptime",
        "free", "nproc", "lscpu", "lsblk", "vmstat", "iostat", "who", "w",
        "ps", "pgrep", "env", "printenv", "which", "type", "sensors",
        "nvidia-smi",
        # network, read-only
        "ss", "netstat", "ping", "dig", "host", "nslookup",
        # tooling, read-only subcommands only (see RESTRICTED_SUBCOMMANDS)
        "git", "systemctl", "journalctl",
    }
)

# For commands whose safety depends on the verb that follows them.
RESTRICTED_SUBCOMMANDS = {
    "git": frozenset(
        {"status", "log", "diff", "show", "branch", "remote", "describe",
         "blame", "ls-files", "rev-parse", "shortlog", "tag"}
    ),
    "systemctl": frozenset(
        {"status", "list-units", "list-unit-files", "list-timers", "is-active",
         "is-enabled", "is-failed", "show", "cat"}
    ),
}

# The tokens shlex hands back for shell syntax. Harmless when the command is
# exec'd as an argv — which is what RESTRICTED does — but a command containing
# them would quietly not mean what the model thought, so they are refused.
_SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", "&", ">", ">>", "<", "<<"})
_SUBSTITUTION = re.compile(r"\$\(|`|<\(|\$\{")

# The path prefixes that make up the operating system itself.
_SYSTEM_DIRS = r"(?:etc|boot|usr|bin|sbin|lib|lib64|var/lib|var/log)"

# Paths that nothing should ever be recursively deleted from.
_ROOTISH = (
    r"(?:/|/\*|/(?:etc|bin|sbin|usr|lib|lib64|boot|dev|proc|sys|var|home|root"
    r"|srv|opt)(?:/\*)?)"
)

# Where a command name can start: the beginning of the line, or after something
# that ends the previous one. Without this, every rule below matches its own
# name inside an argument too — `grep -r 'mount' notes` is not a mount, and
# `echo "rm -rf /"` does not delete anything. It errs the other way instead: a
# `;` or `|` inside a quoted string reads as a command boundary, so a command
# with one of these words quoted after it can still be refused. Wrong-but-safe,
# and rare enough to be worth the trade.
_CMD_START = r"(?:^|[;|&(`{]\s*|\bsudo\s+|\bxargs\s+(?:-\S+\s+)*|\btime\s+)"


def _rule(pattern: str, reason: str) -> tuple[re.Pattern, str]:
    return re.compile(pattern), reason


# Refused at every level, UNRESTRICTED included. The common thread is that
# there is no undo and nobody to ask afterwards.
_CRITICAL_RULES = [
    _rule(
        rf"{_CMD_START}rm\b(?:\s+-\S+)*\s+-\S*[rR]\S*(?:\s+-\S+)*\s+{_ROOTISH}(?:\s|$)",
        "recursively deletes a system directory",
    ),
    _rule(
        rf"{_CMD_START}rm\b(?:\s+-\S+)*\s+(?:~|\$HOME)(?:/\*)?(?:\s|$)",
        "deletes the home directory",
    ),
    _rule(
        rf"{_CMD_START}rm\b[^\n]*{re.escape(str(PROJECT_ROOT))}",
        "deletes Amadeus's own installation",
    ),
    _rule(r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*:", "is a fork bomb"),
    _rule(rf"{_CMD_START}mkfs(?:\.\w+)?\b", "formats a filesystem"),
    _rule(
        rf"{_CMD_START}(?:wipefs|shred)\b[^\n]*\s/dev/", "destroys a block device"
    ),
    _rule(
        rf"{_CMD_START}dd\b[^\n]*\bof=\s*/dev/(?:sd|nvme|vd|hd|mmcblk|disk)",
        "writes straight to a disk device",
    ),
    _rule(r">\s*/dev/(?:sd|nvme|vd|hd|mmcblk)", "writes straight to a disk device"),
    _rule(rf"{_CMD_START}(?:fdisk|sfdisk|sgdisk|parted)\b", "repartitions a disk"),
    _rule(
        rf"{_CMD_START}(?:shutdown|reboot|poweroff|halt)\b",
        "powers off or reboots the host",
    ),
    _rule(
        rf"{_CMD_START}systemctl\s+(?:--?\S+\s+)*"
        r"(?:poweroff|reboot|halt|suspend|hibernate|kexec)\b",
        "powers off or reboots the host",
    ),
    _rule(rf"{_CMD_START}init\s+[06]\b", "changes the runlevel (halt or reboot)"),
    _rule(
        rf"{_CMD_START}ch(?:mod|own)\b(?:\s+-\S+)*\s+-\S*R\S*[^\n]*\s{_ROOTISH}(?:\s|$)",
        "recursively rewrites ownership or permissions of a system directory",
    ),
    _rule(
        rf"{_CMD_START}kill(?:all)?\s+(?:-\S+\s+)*(?:-1\s|1\s|1$)", "kills PID 1"
    ),
]

# Refused at MEDIUM, and so also at RESTRICTED. The common thread is
# administering the machine rather than working on it.
_MEDIUM_RULES = [
    _rule(
        r"(?:^|[;|&(`{]\s*)(?:sudo|doas|pkexec)\b",
        "escalates privileges (sudo/doas/pkexec)",
    ),
    _rule(r"(?:^|[;|&(`{]\s*)su(?:\s|$)", "switches user"),
    _rule(
        rf"{_CMD_START}(?:apt|apt-get|aptitude|dpkg|snap|yum|dnf|pacman|zypper)\s+"
        r"(?:-\S+\s+)*(?:install|remove|purge|upgrade|dist-upgrade|autoremove)",
        "installs or removes system packages",
    ),
    _rule(
        rf"{_CMD_START}systemctl\s+(?:--?\S+\s+)*"
        r"(?:start|stop|restart|reload|enable|disable|mask|unmask|edit|set-property)\b",
        "changes system services",
    ),
    _rule(
        rf"{_CMD_START}service\s+\S+\s+(?:start|stop|restart|reload)\b",
        "changes system services",
    ),
    _rule(
        rf"{_CMD_START}(?:useradd|userdel|usermod|groupadd|groupdel|gpasswd|passwd"
        r"|chpasswd|visudo|adduser|deluser)\b",
        "manages user accounts",
    ),
    _rule(
        rf"{_CMD_START}(?:iptables|ip6tables|nft|ufw|firewall-cmd)\b",
        "changes the firewall",
    ),
    _rule(
        rf"{_CMD_START}(?:mount|umount|swapon|swapoff)\b",
        "mounts or unmounts filesystems",
    ),
    _rule(
        rf"{_CMD_START}(?:docker|podman|kubectl|lxc)\b",
        "controls containers, which on this host is equivalent to root",
    ),
    _rule(
        rf"{_CMD_START}(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k|da|fi)?sh\b",
        "pipes a download straight into a shell",
    ),
    _rule(r"/dev/tcp/", "opens a raw socket from the shell (a reverse shell)"),
    _rule(
        rf"{_CMD_START}(?:nc|ncat|netcat)\b[^\n]*\s-\S*[ec]\b",
        "opens a reverse shell",
    ),
    _rule(
        rf"(?:{_CMD_START}tee\b[^\n]*|>>?)\s*/{_SYSTEM_DIRS}/",
        "writes into a system directory",
    ),
    _rule(
        rf"{_CMD_START}(?:rm|mv|cp|truncate|install|ln)\b[^\n]*\s/{_SYSTEM_DIRS}/",
        "modifies a system directory",
    ),
    _rule(rf"{_CMD_START}crontab\s+-r\b", "wipes the crontab"),
]


class TerminalError(Exception):
    """A command the policy refused, or a session that is not there. Reported to
    the model as an ordinary "Error: ..." string, like every other refusal."""


def _tool(fn):
    """Same contract as workspace_tool._tool: the model always gets a string
    back, and an unexpected failure is reported rather than unwinding into the
    tool-call loop."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TerminalError as e:
            return f"Error: {e}"
        except OSError as e:
            return f"Error: {fn.__name__} failed: {e}"

    return wrapper


def _deny(command: str, rules) -> str | None:
    for pattern, reason in rules:
        if pattern.search(command):
            return reason
    return None


def check_command(command: str, level: AccessLevel | None = None) -> None:
    """Raise TerminalError if `command` is not allowed at `level`.

    Kept apart from the tools that call it so the policy can be tested on its
    own, and so every entry point — one-shot, session, sent keystrokes — goes
    through exactly the same check.
    """
    level = ACCESS_LEVEL if level is None else level
    command = command.strip()

    if not command:
        raise TerminalError("the command is empty.")
    if "\x00" in command:
        raise TerminalError("the command contains a null byte.")

    reason = _deny(command, _CRITICAL_RULES)
    if reason is not None:
        raise TerminalError(
            f"refused at every access level: this command {reason}. There is no "
            f"way to undo it, so it is not available to you."
        )

    if level <= AccessLevel.MEDIUM:
        reason = _deny(command, _MEDIUM_RULES)
        if reason is not None:
            raise TerminalError(
                f"refused at {level.name} access: this command {reason}. Ask "
                f"Aaron to run it, or to raise ACCESS_LEVEL."
            )

    if level is not AccessLevel.RESTRICTED:
        return

    # RESTRICTED: no shell syntax, and an allowlist of read-only commands.
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        raise TerminalError(f"the command could not be parsed ({e}).") from e
    if not tokens:
        raise TerminalError("the command is empty.")
    if _SHELL_OPERATORS.intersection(tokens) or _SUBSTITUTION.search(command):
        raise TerminalError(
            "RESTRICTED access runs one plain command at a time, with no pipes, "
            "redirection, chaining or command substitution. Split it up, or read "
            "the file and do the filtering yourself."
        )

    name = os.path.basename(tokens[0])
    if name not in RESTRICTED_COMMANDS:
        raise TerminalError(
            f"'{name}' is not allowed at RESTRICTED access, which permits only "
            f"read-only inspection commands: "
            f"{', '.join(sorted(RESTRICTED_COMMANDS))}."
        )

    allowed_subs = RESTRICTED_SUBCOMMANDS.get(name)
    if allowed_subs is not None:
        sub = next((t for t in tokens[1:] if not t.startswith("-")), None)
        if sub not in allowed_subs:
            raise TerminalError(
                f"at RESTRICTED access '{name}' is limited to its read-only "
                f"subcommands: {', '.join(sorted(allowed_subs))}."
            )


def describe_access() -> str:
    """One paragraph naming the current level and what it means — for the system
    prompt, so the model knows the rules before it runs into them."""
    blurbs = {
        AccessLevel.RESTRICTED: (
            "read-only inspection only: one plain command at a time, from a fixed "
            "allowlist, no shell syntax, and only inside the workspace"
        ),
        AccessLevel.MEDIUM: (
            "a full shell anywhere on the filesystem, but nothing that administers "
            "the host — no sudo, package installs, service, user, firewall or "
            "mount changes, and no writing into system directories"
        ),
        AccessLevel.UNRESTRICTED: (
            "a full shell with no restrictions beyond a short list of irreversible "
            "commands (wiping /, formatting a disk, powering off) that are refused "
            "at every level"
        ),
    }
    return (
        f"Terminal access is set to {ACCESS_LEVEL.name}: {blurbs[ACCESS_LEVEL]}. "
        f"Commands run on the Ubuntu host as the user Amadeus runs as, starting in "
        f"{WORKSPACE_ROOT}."
    )


# ---------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------

# Escape sequences and carriage returns a pty leaves in the stream, stripped so
# the model reads output rather than cursor movements.
_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|\r"
)


# The end marker terminal_run appends to every command. A command that outruns
# its timeout keeps going, so its marker turns up later, in the middle of
# somebody else's terminal_read — every reader rewrites it into something the
# model can act on rather than leaving a raw uuid in the output.
_MARKER = re.compile(r"\n?__AMTH_[0-9a-f]{32}__(\d+)\n?")


def _clean(text: str) -> str:
    return _ANSI.sub("", text)


def _mark_finished(text: str) -> str:
    return _MARKER.sub(lambda m: f"\n[the command finished, exit {m.group(1)}]", text)


def _truncate(text: str) -> str:
    """Keep the head and the tail — the start says what happened, the end says
    how it ended, and the middle of a long dump is rarely the interesting part."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    cut = len(text) - MAX_OUTPUT_CHARS
    return (
        f"{text[:half]}\n\n... [{cut} characters cut from the middle] ...\n\n"
        f"{text[-half:]}"
    )


def _child_env() -> dict:
    """The environment commands run in.

    TERM=dumb and the pager settings matter more than they look: a command that
    opens a pager (git log, systemctl status, man) would otherwise sit waiting
    for a keypress that is never coming, and burn the whole timeout.
    """
    env = os.environ.copy()
    env.update(
        {
            "TERM": "dumb",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "SYSTEMD_PAGER": "cat",
            "DEBIAN_FRONTEND": "noninteractive",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _resolve_cwd(cwd: str | None) -> Path:
    """Relative paths hang off the workspace root, the same way the file tools
    treat them. RESTRICTED cannot leave it at all."""
    if not cwd:
        return WORKSPACE_ROOT
    path = Path(cwd)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path = path.resolve()

    if ACCESS_LEVEL is AccessLevel.RESTRICTED and not path.is_relative_to(
        WORKSPACE_ROOT
    ):
        raise TerminalError(
            f"at RESTRICTED access commands run inside the workspace "
            f"({WORKSPACE_ROOT}); '{cwd}' is outside it."
        )
    if not path.is_dir():
        raise TerminalError(f"'{cwd}' is not a directory.")
    return path


def _clamp_timeout(timeout: int | None) -> int:
    ceiling = MAX_TIMEOUT[ACCESS_LEVEL]
    if timeout is None:
        return min(DEFAULT_TIMEOUT, ceiling)
    return max(1, min(int(timeout), ceiling))


def _require_shell() -> None:
    if not SHELL:
        raise TerminalError(
            "bash was not found on this host — these tools need a POSIX shell."
        )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command and anything it started. Killing only the direct child
    leaves the other half of a pipeline running and still holding the pipe."""
    if proc.poll() is not None:
        return
    try:
        if _POSIX:
            group = os.getpgid(proc.pid)
            os.killpg(group, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                os.killpg(group, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


@_tool
def run_command(command: str, cwd: str = None, timeout: int = None) -> str:
    """Run one command, wait for it to finish, and return its output."""
    print(f"[terminal] run_command: {command!r} (cwd={cwd or '.'})")
    _require_shell()
    check_command(command)
    directory = _resolve_cwd(cwd)
    limit = _clamp_timeout(timeout)

    # RESTRICTED is allowlistable precisely because it never reaches a shell:
    # the argv is exec'd as it stands, so there is no quoting, globbing or
    # expansion to get wrong. Every other level wants a real shell.
    if ACCESS_LEVEL is AccessLevel.RESTRICTED:
        argv = shlex.split(command)
    else:
        argv = [SHELL, "-c", command]

    options = {
        "cwd": str(directory),
        "env": _child_env(),
        "stdin": subprocess.DEVNULL,  # a prompt should fail, not hang
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,  # interleaved, the way a terminal shows it
        "text": True,
        "errors": "replace",
    }
    if _POSIX:
        options["start_new_session"] = True  # gives _kill_tree a process group

    started = time.monotonic()
    try:
        proc = subprocess.Popen(argv, **options)
    except FileNotFoundError:
        return f"Error: '{argv[0]}' was not found on this host."

    timed_out = False
    try:
        output = proc.communicate(timeout=limit)[0]
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        output = proc.communicate()[0] or ""
    elapsed = time.monotonic() - started

    output = _truncate(_clean(output).strip())
    header = f"$ {command}\n[{directory}]"
    if timed_out:
        return (
            f"{header}\nKilled after {limit}s — it had not finished. Output so "
            f"far:\n{output or '(none)'}\n\nIf it needs longer, start a session "
            f"with terminal_open and run it there, where it keeps going between "
            f"calls."
        )
    return (
        f"{header}\nexit {proc.returncode} in {elapsed:.1f}s\n"
        f"{output or '(no output)'}"
    )


# ---------------------------------------------------------------------------
# Persistent sessions
# ---------------------------------------------------------------------------


class _Session:
    """A bash process on a pty, plus a thread draining it into a buffer.

    The thread is the point: with nothing reading the master end continuously, a
    command that prints more than the pty buffer holds blocks forever, and the
    session is dead with no way to tell why.
    """

    def __init__(self, name: str, cwd: Path):
        self.name = name
        self.cwd = cwd
        self.lock = threading.Lock()
        self.buffer = bytearray()

        master, slave = pty.openpty()
        # A wide, tall window (50 rows x 200 cols): bash wraps its output to the
        # terminal size, and output hard-wrapped at column 80 is miserable to
        # read back.
        try:
            fcntl.ioctl(
                slave, termios.TIOCSWINSZ, b"\x00\x32\x00\xc8\x00\x00\x00\x00"
            )
        except OSError:
            pass

        self.proc = subprocess.Popen(
            [SHELL, "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=str(cwd),
            env=_child_env(),
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)
        self.master = master
        self.closed = False

        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self.master, 4096)
            except OSError:  # the pty closed under us — the shell exited
                break
            if not data:
                break
            with self.lock:
                self.buffer.extend(data)

    def write(self, text: str) -> None:
        if self.closed or self.proc.poll() is not None:
            raise TerminalError(
                f"session '{self.name}' has exited. Open a new one with "
                f"terminal_open."
            )
        os.write(self.master, text.encode())

    def peek(self) -> str:
        with self.lock:
            return bytes(self.buffer).decode("utf-8", errors="replace")

    def take(self) -> str:
        """Read and clear everything received so far."""
        with self.lock:
            data = bytes(self.buffer)
            self.buffer.clear()
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            os.close(self.master)
        except OSError:
            pass


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


@atexit.register
def _close_all_sessions() -> None:
    """Don't leave shells running on the host after the agent goes away."""
    for session in list(_sessions.values()):
        session.close()
    _sessions.clear()


def _require_pty() -> None:
    _require_shell()
    if not _PTY_AVAILABLE:
        raise TerminalError(
            "terminal sessions need a POSIX pty, which this host does not have. "
            "Use run_command instead."
        )


def _get(name: str) -> _Session:
    session = _sessions.get(name)
    if session is None:
        known = ", ".join(sorted(_sessions)) or "none"
        raise TerminalError(f"there is no session named '{name}'. Open: {known}.")
    if session.proc.poll() is not None:
        raise TerminalError(
            f"session '{name}' has exited (status {session.proc.returncode}). "
            f"Close it with terminal_close and open a new one."
        )
    return session


@_tool
def terminal_open(name: str = "main", cwd: str = None) -> str:
    """Start a persistent bash session and leave it running."""
    print(f"[terminal] terminal_open: session '{name}' (cwd={cwd or '.'})")
    _require_pty()
    existing = _sessions.get(name)
    if existing is not None and existing.proc.poll() is None:
        return f"Session '{name}' is already open."
    if len(_sessions) >= MAX_SESSIONS:
        raise TerminalError(
            f"there are already {MAX_SESSIONS} sessions open "
            f"({', '.join(sorted(_sessions))}). Close one first."
        )

    directory = _resolve_cwd(cwd)
    with _sessions_lock:
        dead = _sessions.pop(name, None)
        if dead is not None:
            dead.close()
        session = _Session(name, directory)
        _sessions[name] = session

    # Quiet the shell down before anything else runs in it: echo off, so the
    # command is not read back as part of its own output; no prompt string, so
    # there is nothing to parse around; no history, so a long session does not
    # rewrite .bash_history.
    session.write(
        "stty -echo 2>/dev/null; export PS1='' PS2='' PROMPT_COMMAND=''; "
        "unset HISTFILE; set +o history\n"
    )
    time.sleep(0.4)
    session.take()  # discard the login banner and the setup echo
    return (
        f"Opened session '{name}' in {directory}. It keeps its working directory "
        f"and environment between calls — run things in it with terminal_run."
    )


@_tool
def terminal_run(command: str, name: str = "main", timeout: int = None) -> str:
    """Run a command in a session and wait for it to finish."""
    print(f"[terminal] terminal_run: {command!r} in session '{name}'")
    _require_pty()
    check_command(command)
    if "\n" in command.strip():
        raise TerminalError(
            "a session takes one line at a time. Join the steps with '&&', or "
            "write a script into the workspace and run that."
        )
    session = _get(name)
    limit = _clamp_timeout(timeout)

    # A shell on a pty gives no reliable sign that a command has finished, so
    # the command prints its own end marker and exit status. The marker is a
    # fresh uuid every time, so output that happens to contain the word is never
    # mistaken for the real one.
    marker = f"__AMTH_{uuid.uuid4().hex}__"
    session.take()  # whatever is left from before is not this command's output
    session.write(f"{command}; printf '\\n{marker}%s\\n' \"$?\"\n")

    pattern = re.compile(re.escape(marker) + r"(\d+)")
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        pending = _clean(session.peek())
        match = pattern.search(pending)
        if match:
            session.take()
            output = _truncate(_mark_finished(pending[: match.start()]).strip())
            return f"$ {command}\nexit {match.group(1)}\n{output or '(no output)'}"
        time.sleep(0.05)

    # Still running. Unlike run_command this is not a failure — the command
    # carries on in the session, and the model can watch it or stop it.
    partial = _truncate(_mark_finished(_clean(session.peek())).strip())
    return (
        f"$ {command}\nStill running after {limit}s. Output so far:\n"
        f"{partial or '(none)'}\n\nIt is still going in session '{name}' — call "
        f"terminal_read to see more, or terminal_interrupt to stop it."
    )


@_tool
def terminal_read(name: str = "main", wait: int = 2) -> str:
    """Read whatever a session has printed since it was last read."""
    print(f"[terminal] terminal_read: session '{name}' (wait={wait}s)")
    _require_pty()
    session = _get(name)
    deadline = time.monotonic() + max(0, min(int(wait), 60))
    while time.monotonic() < deadline and not session.peek():
        time.sleep(0.05)
    output = _truncate(_mark_finished(_clean(session.take())).strip())
    return output or f"Session '{name}' has printed nothing new."


@_tool
def terminal_send_keys(text: str, name: str = "main", enter: bool = True) -> str:
    """Type into a session without waiting for a command to finish — for
    answering a program that is sitting at a prompt."""
    print(f"[terminal] terminal_send_keys: {text!r} to session '{name}' (enter={enter})")
    _require_pty()
    if ACCESS_LEVEL is AccessLevel.RESTRICTED:
        raise TerminalError(
            "sending keystrokes is not available at RESTRICTED access, which runs "
            "one complete read-only command at a time."
        )
    session = _get(name)

    # The denylists still apply: if nothing is waiting for input, whatever is
    # typed here lands on the shell's command line like any other command. The
    # RESTRICTED allowlist deliberately does not, since a bare "y" answering a
    # prompt is not a command and would never pass it.
    reason = _deny(text, _CRITICAL_RULES) or (
        _deny(text, _MEDIUM_RULES) if ACCESS_LEVEL <= AccessLevel.MEDIUM else None
    )
    if reason is not None:
        raise TerminalError(f"refused: this input {reason}.")

    session.write(text + ("\n" if enter else ""))
    time.sleep(0.5)
    output = _truncate(_mark_finished(_clean(session.take())).strip())
    return output or f"Sent to '{name}'. Nothing printed back yet."


@_tool
def terminal_interrupt(name: str = "main") -> str:
    """Send Ctrl-C to a session to stop whatever is running in it."""
    print(f"[terminal] terminal_interrupt: session '{name}'")
    _require_pty()
    session = _get(name)
    session.write("\x03")
    time.sleep(0.4)
    output = _truncate(_mark_finished(_clean(session.take())).strip())
    return f"Sent Ctrl-C to '{name}'.\n{output}".strip()


@_tool
def terminal_list() -> str:
    """List the open terminal sessions."""
    print("[terminal] terminal_list")
    if not _sessions:
        return "No terminal sessions are open."
    rows = []
    for name, session in sorted(_sessions.items()):
        state = "running" if session.proc.poll() is None else "exited"
        rows.append(f"{name}: {state}, started in {session.cwd}")
    return "\n".join(rows)


@_tool
def terminal_close(name: str = "main") -> str:
    """Close a session and kill anything still running in it."""
    print(f"[terminal] terminal_close: session '{name}'")
    with _sessions_lock:
        session = _sessions.pop(name, None)
    if session is None:
        raise TerminalError(f"there is no session named '{name}'.")
    session.close()
    return f"Closed session '{name}'."


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# The level is fixed at import time, so the descriptions say what is actually
# allowed rather than describing the widest case and letting the model find out
# by being refused.
_LEVEL_NOTE = {
    AccessLevel.RESTRICTED: (
        "Access is RESTRICTED: one plain read-only command at a time, no pipes, "
        "redirection or chaining, and only inside the workspace. Allowed "
        "commands: " + ", ".join(sorted(RESTRICTED_COMMANDS)) + "."
    ),
    AccessLevel.MEDIUM: (
        "Access is MEDIUM: a full shell, pipes and redirection included, anywhere "
        "on the filesystem. Refused: sudo, package installs, service, user, "
        "firewall and mount changes, containers, and writes into system "
        "directories."
    ),
    AccessLevel.UNRESTRICTED: (
        "Access is UNRESTRICTED: a full shell with no restrictions beyond a short "
        "list of irreversible commands (wiping /, formatting a disk, powering off "
        "the host), which are always refused."
    ),
}[ACCESS_LEVEL]

_SESSION_ARG = {
    "type": "string",
    "description": "Name of the session. Defaults to 'main'.",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Runs a single shell command on the Ubuntu host, waits for it to "
                "finish, and returns its output and exit status. Use this for "
                "one-off commands; use terminal_open and terminal_run instead "
                "when a command needs the working directory or environment left "
                "behind by an earlier one, or when it will take a while. "
                + _LEVEL_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command line to run.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to run in. Relative paths resolve from the "
                            "workspace root. Omit for the workspace root."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to allow before the command is killed. "
                            f"Defaults to {DEFAULT_TIMEOUT}, capped at "
                            f"{MAX_TIMEOUT[ACCESS_LEVEL]}."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_open",
            "description": (
                "Opens a persistent bash terminal session on the host and leaves "
                "it running. The session keeps its working directory, its "
                "environment variables and any program still running in it "
                "between calls, so 'cd' and exports stick."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "A name to refer to this session by later. Defaults "
                            "to 'main'."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to start in. Relative paths resolve from "
                            "the workspace root. Omit for the workspace root."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_run",
            "description": (
                "Runs one command in an open terminal session and returns its "
                "output and exit status. If it has not finished when the timeout "
                "is up it keeps running in the session rather than being killed — "
                "follow it with terminal_read. " + _LEVEL_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The command line to run. One line — join steps with "
                            "'&&' rather than newlines."
                        ),
                    },
                    "name": _SESSION_ARG,
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to wait for it to finish before returning "
                            f"and leaving it running. Defaults to "
                            f"{DEFAULT_TIMEOUT}, capped at "
                            f"{MAX_TIMEOUT[ACCESS_LEVEL]}."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_read",
            "description": (
                "Returns whatever a session has printed since the last read — the "
                "way to follow a command that is still running."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": _SESSION_ARG,
                    "wait": {
                        "type": "integer",
                        "description": (
                            "Seconds to wait for output before giving up. "
                            "Defaults to 2."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_send_keys",
            "description": (
                "Types text into a session without waiting for a command to "
                "finish — for answering a program sitting at a prompt (a "
                "confirmation, a menu choice). For ordinary commands use "
                "terminal_run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to type."},
                    "name": _SESSION_ARG,
                    "enter": {
                        "type": "boolean",
                        "description": (
                            "Whether to press Enter afterwards. Defaults to true."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_interrupt",
            "description": (
                "Sends Ctrl-C to a session, stopping whatever is running in it "
                "while leaving the session itself open."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": _SESSION_ARG},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_list",
            "description": "Lists the terminal sessions that are currently open.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_close",
            "description": (
                "Closes a terminal session and kills anything still running in it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": _SESSION_ARG},
                "required": [],
            },
        },
    },
]


FUNCTIONS = {
    "run_command": run_command,
    "terminal_open": terminal_open,
    "terminal_run": terminal_run,
    "terminal_read": terminal_read,
    "terminal_send_keys": terminal_send_keys,
    "terminal_interrupt": terminal_interrupt,
    "terminal_list": terminal_list,
    "terminal_close": terminal_close,
}
