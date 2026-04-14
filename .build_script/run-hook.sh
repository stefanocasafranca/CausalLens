#!/bin/bash
# build_script project-level hook — v18.0
# Zero-install: fires for ANY Claude Code session in this project.
# Uses dirname "" to self-locate minimal-hook.py regardless of clone path.
#
# IMPORTANT: If the user-level plugin is installed, the plugin's own hook ALREADY
# fires for this session (user-level hooks are always active). Exit immediately to
# prevent double-injection of MANDATORY-SYSTEM-TASK.
# Only run minimal-hook.py when the plugin is NOT installed.
SCRIPT_DIR="$(cd "$(dirname "")" && pwd)"
PLUGIN="$HOME/.claude/plugins/local/build_script/hooks/user-prompt-handler.py"
MINIMAL="$SCRIPT_DIR/minimal-hook.py"
if [ -f "$PLUGIN" ]; then
    # Plugin installed — user-level hook handles this. Silent no-op.
    exit 0
elif [ -f "$MINIMAL" ]; then
    exec python3 "$MINIMAL" 2>/dev/null || exec python "$MINIMAL" 2>/dev/null || exit 0
else
    exit 0
fi
