<!-- .github/copilot-instructions.md -->
# Copilot / AI-agent instructions for this workspace

This repository is a small, local Python environment setup notebook. The instructions below are focused and actionable so an AI coding agent can be immediately productive when editing, testing, or improving files here.

**Quick orientation**
- **Primary artifact:** `pythonsetup.ipynb` — a sequence of cells that verify Python, package installations, and basic git/SSH setup. Treat the notebook as an operational bootstrap script rather than a finished application.
- **No CI or test harness found:** there are only interactive cells; do not assume any automated tests or pipelines exist unless the user adds them.

**What the workspace expects (patterns & examples)**
- Cell 1 (notebook start): lightweight sanity check script `simple_sum.py` style example used to confirm Python runtime.
- Cells that check packages: multiple cells check/import modules (pandas, numpy, matplotlib, sklearn). Example check pattern:
  - try: import pandas; print(pandas.__version__)
- Several cells contain raw shell commands (plain text lines like `pip install pandas`, `git config --global ...`, `ssh-keygen ...`, `pip list`). These are not executed by default in the notebook: treat them as instructions to be run in a terminal or transformed into notebook-appropriate calls.

**Agent behavior rules (must follow)**
- Do not run commands that change global user settings or create credentials without explicit confirmation from the user. Specifically: `git config --global ...` and `ssh-keygen` must be run only after asking.
- When converting or executing shell commands found in notebook cells, prefer one of these safe approaches:
  - Convert to explicit notebook shell invocations (prepend `!` or use `%%bash`) when the goal is to execute in the notebook UI.
  - Or convert the plain `pip install ...` lines into recommended, reproducible commands such as `python -m pip install --upgrade pandas` and document whether it should be run inside a virtual environment.
- Preserve personalized values (e.g., `user.name` and `user.email`) found in notebook cells — do not overwrite them unless the user asks.
- When suggesting package installs, prefer `python -m pip install ...` and include a comment about virtual environments (venv/conda) because the notebook shows direct `pip` usage.

**Concrete examples from this repo (use these when editing or suggesting changes)**
- Replace a plain-line `pip install pandas` (Cell 4) with `!python -m pip install --upgrade pandas` if executing inside the notebook.
- When recommending headless execution for verification, suggest: `jupyter nbconvert --to notebook --execute pythonsetup.ipynb --ExecutePreprocessor.timeout=600` (explain that shell commands embedded as plain text will not run unless converted).
- For listing installed packages in a non-interactive script, prefer `python -m pip list` (Cell 10 currently contains `pip list`).

**Developer workflow and useful commands**
- Open and run the notebook in VS Code's Jupyter UI to step through checks interactively.
- To execute the notebook end-to-end (CI-like):
  - `python -m pip install -r requirements.txt` (if a `requirements.txt` is later added)
  - `jupyter nbconvert --to notebook --execute pythonsetup.ipynb`
- To install packages safely from an agent suggestion: recommend `python -m pip install --upgrade <pkg>` and mention using a virtual env: `python -m venv .venv && source .venv/bin/activate` (macOS zsh).

**When modifying the notebook**
- If adding shell commands, use notebook magics or `!` for clarity and execution reproducibility.
- If turning installation steps into a reproducible script, create a `requirements.txt` or `environment.yml` and update the README rather than leaving ad-hoc `pip` lines in code cells.

**What not to do**
- Do not run or auto-commit changes that alter the repository owner's global git config or generate SSH keys.
- Do not assume tests or CI will validate changes — add a minimal validation instruction when proposing changes (e.g., run the version-check cells locally).

If anything in these instructions is unclear or you'd like more detail (for example, a suggested `requirements.txt` or turning notebook checks into a script), tell me which parts you'd like expanded.
