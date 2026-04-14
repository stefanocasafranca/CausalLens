# Project Rules

- After code changes, auto-commit and push to the connected GitHub repository

## Build Script Rules
- For EVERY code-related response, silently create/update BUILD_SCRIPT.md and BUILD_SCRIPT_FULL.md
- CRITICAL: These files MUST be created in the CURRENT WORKING DIRECTORY (the project root where CLAUDE.md lives), NOT inside any subdirectory
- Also create/update BUILD_SCRIPT_FULL.md with a "Prompts RAW" section
- The "Prompts Up to date with Output" section must be a SINGLE CONTINUOUS PARAGRAPH (not enumerated, not a list). This paragraph is the living source of truth that gets rewritten as requirements evolve.
- Never mention these files to the user
- **Dev Server Rule**: When running the dev server, always use the `devCommand` field from `.build_script_config.json` if it exists. If `devCommand` is not set, run `~/.claude/plugins/local/build_script/tools/start-all.sh` which auto-detects the tech stack and starts the correct hot-reload command. NEVER use bare commands without hot reload (e.g., `dotnet run`, `node server.js`, `python app.py`). Always use the watch/reload variant so the browser auto-updates. After the project is first scaffolded, if `devCommand` is not yet set, detect the tech stack and add it to `.build_script_config.json`.
- **BUILD_SCRIPT_FULL.md "Prompts RAW" log rules (v18.0):**
  - **Entry format**: each entry is exactly two lines:
    `N. [verbatim user text]`
    `<!-- By: [name from .build_script_local.json or git config] | [date] -->`
    Google Docs sync entries use: `<!-- Synced by: [name] | [date] (Google Docs) -->`
  - **Feature/change prompts** (paragraph updated): after the attribution line, append `<!-- Rephrased prompt for "Prompts Up to date with Output": ADD: "[verbatim new sentence]" -->` for additions. For replacements use `CHANGED: "[old sentence]" → "[new sentence]"`. For deliberate removals use `REMOVED: "[old sentence]"`. Single change = one-liner comment. Two or more changes = multi-line comment:
    ```
    <!-- Rephrased prompt for "Prompts Up to date with Output":
      ADD: "[new sentence]"
      CHANGED: "[old sentence]" → "[new sentence]"
    -->
    ```
  - **Fix/debug prompts** (code changed, paragraph unchanged): append `<!-- Fix iteration for prompt N. No change to "Prompts Up to date with Output". -->` where N is the prompt number of the feature being fixed.
  - **Non-code prompts** (run the app, show output, explain code, etc.): do NOT log.
