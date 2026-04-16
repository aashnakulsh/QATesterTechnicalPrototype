# QA Tester Technical Prototype

A technical prototype for an agentic QA workflow. The agent interacts with a test web app, performs a bounded test flow, detects visible failures, and drafts reproducible bug reports.

## Files in this repo

* `toy_app.html`
  Small test app with planted bugs.

* `toy_app_complex.html`
  Richer test app with search, filters, modal, cart, coupon, shipping estimate, checkout, and settings.

* `agent_runner_local.py`
  Free local runner with rule-based logic. No API credits needed.

* `agent_runner_ollama.py`
  Real local LLM runner for the simple app.

* `agent_runner_ollama_complex.py`
  Real local LLM runner for the richer app, with fallback logic if Ollama fails.

* `agent_runner.py`
  OpenAI API version. Requires API billing/credits.

## Recommended setup order

For the smoothest experience:

1. install Python dependencies
2. start the local test app
3. run `agent_runner_local.py` first
4. then try the Ollama versions

## 1. Clone the repo

```bash
git clone https://github.com/aashnakulsh/QATesterTechnicalPrototype.git
cd QATesterTechnicalPrototype
```

## 2. Install Python dependencies

### Windows (Command Prompt)

```bat
py -m pip install openai playwright ollama pydantic
py -m playwright install
```

If `py` does not work, try:

```bat
python -m pip install openai playwright ollama pydantic
python -m playwright install
```

## 3. Start the local test app

In one terminal window, from the repo folder:

```bat
py -m http.server 8000
```

This serves the HTML files locally at:

```text
http://127.0.0.1:8000
```

## 4. Run the free local version first

Open a second terminal in the same repo folder.

### Simple app

```bat
set APP_URL=http://127.0.0.1:8000/toy_app.html
py agent_runner_local.py
```

### Complex app

```bat
set APP_URL=http://127.0.0.1:8000/toy_app_complex.html
py agent_runner_local.py
```

This version does not need API credits or Ollama.

## 5. Ollama setup

If you want to use the real local LLM versions, you need both:

1. the Ollama app/service installed on your computer
2. the Python package `ollama` installed with pip

### 5a. Install Ollama app

Install Ollama on your machine first.

Then verify it works by running:

```bat
ollama list
```

If that command is not recognized, Ollama itself is not installed correctly yet.

### 5b. Pull a model

We used `qwen3` in this project.

```bat
ollama pull qwen3
```

You can test that the model works with:

```bat
ollama run qwen3
```

Then type something simple like:

```text
hello
```

If it answers, Ollama is working.

## 6. Run the Ollama versions

Open a second terminal in the repo folder.

### Simple Ollama app

```bat
set APP_URL=http://127.0.0.1:8000/toy_app.html
set OLLAMA_MODEL=qwen3
py agent_runner_ollama.py
```

### Complex Ollama app

```bat
set APP_URL=http://127.0.0.1:8000/toy_app_complex.html
set OLLAMA_MODEL=qwen3
py agent_runner_ollama_complex.py
```

### Run the script

```bat
set APP_URL=http://127.0.0.1:8000/toy_app.html
py agent_runner.py
```

If you get a `429 insufficient_quota` error, your API account does not currently have usable billing/quota.

## What each script does

Each runner:

1. opens the test app in a browser
2. reads the visible UI state
3. decides the next action
4. executes that action
5. checks for visible failures
6. saves logs and drafted bug reports

## Output folders

The scripts save logs into folders like:

* `logs_local`
* `logs_ollama`
* `logs_ollama_complex`

These logs are useful for debugging and reviewing the agent’s behavior.

Typical files include:

* `step_01_observation.json`
* `step_01_decision.json`
* `run_history.json`
* `bug_01.json`
* `bug_01.md`

## Troubleshooting

### `playwright` is not recognized

Use:

```bat
py -m playwright install
```

instead of:

```bat
playwright install
```

### `No module named 'ollama'`

Install the Python package:

```bat
py -m pip install ollama pydantic
```

### `ollama` is not recognized

That means the Ollama app itself is not installed correctly yet. Install Ollama first, then reopen Command Prompt.

### Ollama returns blank or invalid JSON

Try:

```bat
ollama list
ollama run qwen3
```

If the model chats normally but the script still fails, try the fallback-friendly script:

```bat
set APP_URL=http://127.0.0.1:8000/toy_app_complex.html
set OLLAMA_MODEL=qwen3
py agent_runner_ollama_complex.py
```

### OpenAI API returns `429 insufficient_quota`

That means the API key is valid, but the API project/account does not currently have usable billing/quota.

### `ls` does not work in Command Prompt

Use:

```bat
dir
```

instead.

## Suggested first run

If you just want to see the system working quickly, do this.

### Terminal 1

```bat
py -m http.server 8000
```

### Terminal 2

```bat
set APP_URL=http://127.0.0.1:8000/toy_app_complex.html
py agent_runner_local.py
```

Then once that works, try:

```bat
set APP_URL=http://127.0.0.1:8000/toy_app_complex.html
set OLLAMA_MODEL=qwen3
py agent_runner_ollama_complex.py
```

## Notes

* This is a bounded prototype, not a production QA platform.
* The test apps include intentional bugs.
* The Ollama versions are real local LLM-driven prototypes.
* The local version is the easiest way to verify the browser-testing pipeline first.
