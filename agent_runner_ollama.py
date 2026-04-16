import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

from ollama import chat
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, ValidationError


APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000/toy_app.html")
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
LOG_DIR = os.getenv("LOG_DIR", "logs_ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")

APP_DESCRIPTION = (
    "This app has a simple shopping flow: Home -> Product -> Cart -> Checkout -> Confirmation. "
    "The QA goal is to verify coupon pricing and ZIP validation. "
    "Expected behavior: coupon SUMMER20 should reduce the $100 price to $80, and invalid ZIP codes should be rejected before order completion."
)

TEST_GOAL = (
    "Walk through the shopping flow, test coupon behavior, test invalid ZIP validation, "
    "and stop when the flow is complete or when no reasonable next action remains."
)

ALLOWED_ACTIONS = [
    "CLICK_TEXT",
    "TYPE_TEXT",
    "CLICK_AND_TYPE_BY_TEXT",
    "DONE",
]

SYSTEM_PROMPT = f"""
You are a cautious QA testing agent.

You receive:
- an app description
- a test goal
- the current observation of the page
- the history of previous actions

Your job:
1. Understand the current state of the app.
2. Choose the next single action that best advances the test.
3. Detect suspicious behavior when visible evidence suggests a bug.
4. Return strict JSON only.

Important rules:
- Do not invent UI elements that are not in the observation.
- Prefer one small, reversible action at a time.
- If an observed output violates the expected behavior, log a suspected bug.
- If the test is complete, return action type DONE.
- Allowed action types are exactly: {', '.join(ALLOWED_ACTIONS)}.

For CLICK_TEXT, include target_text.
For TYPE_TEXT, include field_text and input_text.
For CLICK_AND_TYPE_BY_TEXT, include field_text, input_text, and click_text.
For DONE, leave action targets null.
""".strip()


class ActionModel(BaseModel):
    type: Literal["CLICK_TEXT", "TYPE_TEXT", "CLICK_AND_TYPE_BY_TEXT", "DONE"]
    target_text: Optional[str] = None
    field_text: Optional[str] = None
    input_text: Optional[str] = None
    click_text: Optional[str] = None


class BugModel(BaseModel):
    title: str
    severity: str
    expected: str
    actual: str
    reason: str


class DecisionModel(BaseModel):
    thought: str
    expected_check: str
    action: ActionModel
    bug_suspected: bool
    bug: Optional[BugModel] = None


@dataclass
class Action:
    type: str
    target_text: Optional[str] = None
    field_text: Optional[str] = None
    input_text: Optional[str] = None
    click_text: Optional[str] = None


@dataclass
class BugReport:
    title: str
    severity: str
    expected: str
    actual: str
    reason: str
    reproduction_steps: List[str]


@dataclass
class AgentDecision:
    thought: str
    expected_check: str
    action: Action
    bug_suspected: bool
    bug: Optional[Dict[str, Any]] = None


def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def save_json(filename: str, data: Any) -> None:
    ensure_log_dir()
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_markdown(filename: str, text: str) -> None:
    ensure_log_dir()
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_observation(page) -> Dict[str, Any]:
    title = page.title()
    url = page.url

    visible_buttons = []
    for locator in [
        page.locator("button:visible"),
        page.locator("a:visible"),
        page.locator("[role='button']:visible"),
    ]:
        count = locator.count()
        for i in range(count):
            try:
                text = locator.nth(i).inner_text().strip()
                if text and text not in visible_buttons:
                    visible_buttons.append(text)
            except Exception:
                pass

    inputs = []
    input_locator = page.locator("input:visible")
    for i in range(input_locator.count()):
        try:
            element = input_locator.nth(i)
            placeholder = element.get_attribute("placeholder") or ""
            name = element.get_attribute("name") or ""
            label = placeholder or name or f"input_{i}"
            value = element.input_value()
            inputs.append({"label": label, "value": value})
        except Exception:
            pass

    body_text = page.locator("body").inner_text(timeout=3000)
    compact_body_text = "\n".join(line.strip() for line in body_text.splitlines() if line.strip())
    compact_body_text = compact_body_text[:3000]

    return {
        "title": title,
        "url": url,
        "buttons": visible_buttons,
        "inputs": inputs,
        "body_text": compact_body_text,
    }


def ask_ollama(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> AgentDecision:
    user_payload = {
        "app_description": APP_DESCRIPTION,
        "test_goal": TEST_GOAL,
        "observation": observation,
        "history": history,
    }

    response = chat(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ],
    format=DecisionModel.model_json_schema(),
    stream=False,
    keep_alive="30m",
    options={
        "temperature": 0,
        "num_predict": 80,
        "num_ctx": 1024,
    },
)

    raw = response.message.content
    parsed = DecisionModel.model_validate_json(raw)

    return AgentDecision(
        thought=parsed.thought,
        expected_check=parsed.expected_check,
        action=Action(**parsed.action.model_dump()),
        bug_suspected=parsed.bug_suspected,
        bug=parsed.bug.model_dump() if parsed.bug else None,
    )


def click_by_text(page, text: str) -> None:
    candidates = [
        page.get_by_role("button", name=text),
        page.get_by_text(text, exact=True),
        page.locator(f"text={text}"),
    ]
    for locator in candidates:
        try:
            locator.first.click(timeout=2000)
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not click element with text: {text}")


def type_into_best_input(page, field_text: str, input_text: str) -> None:
    try:
        page.get_by_placeholder(field_text).fill(input_text, timeout=2000)
        return
    except Exception:
        pass

    try:
        page.locator(f"input[name='{field_text}']").first.fill(input_text, timeout=2000)
        return
    except Exception:
        pass

    field_lower = field_text.lower()
    inputs = page.locator("input:visible")
    for i in range(inputs.count()):
        try:
            element = inputs.nth(i)
            placeholder = (element.get_attribute("placeholder") or "").lower()
            name = (element.get_attribute("name") or "").lower()
            if field_lower in placeholder or field_lower in name:
                element.fill(input_text, timeout=2000)
                return
        except Exception:
            pass

    raise RuntimeError(f"Could not find input for field: {field_text}")


def execute_action(page, action: Action) -> str:
    if action.type == "DONE":
        return "Agent marked flow complete."

    if action.type == "CLICK_TEXT":
        if not action.target_text:
            raise ValueError("CLICK_TEXT requires target_text")
        click_by_text(page, action.target_text)
        return f"Clicked '{action.target_text}'."

    if action.type == "TYPE_TEXT":
        if not action.field_text or action.input_text is None:
            raise ValueError("TYPE_TEXT requires field_text and input_text")
        type_into_best_input(page, action.field_text, action.input_text)
        return f"Typed '{action.input_text}' into '{action.field_text}'."

    if action.type == "CLICK_AND_TYPE_BY_TEXT":
        if not action.field_text or action.input_text is None or not action.click_text:
            raise ValueError("CLICK_AND_TYPE_BY_TEXT requires field_text, input_text, and click_text")
        type_into_best_input(page, action.field_text, action.input_text)
        click_by_text(page, action.click_text)
        return f"Typed '{action.input_text}' into '{action.field_text}' and clicked '{action.click_text}'."

    raise ValueError(f"Unknown action type: {action.type}")


def build_bug_report(decision: AgentDecision, action_history: List[str]) -> Optional[BugReport]:
    if not decision.bug_suspected or not decision.bug:
        return None

    bug = decision.bug
    return BugReport(
        title=bug.get("title", "Untitled bug"),
        severity=bug.get("severity", "Medium"),
        expected=bug.get("expected", ""),
        actual=bug.get("actual", ""),
        reason=bug.get("reason", ""),
        reproduction_steps=action_history.copy(),
    )


def bug_report_to_markdown(report: BugReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"**Severity:** {report.severity}",
        "",
        "## Expected",
        report.expected,
        "",
        "## Actual",
        report.actual,
        "",
        "## Reason",
        report.reason,
        "",
        "## Steps to Reproduce",
    ]
    for idx, step in enumerate(report.reproduction_steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    history: List[Dict[str, Any]] = []
    action_history: List[str] = []
    bug_reports: List[BugReport] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(APP_URL)

        for step_idx in range(1, MAX_STEPS + 1):
            try:
                page.wait_for_timeout(500)
                observation = build_observation(page)
                save_json(f"step_{step_idx:02d}_observation.json", observation)

                decision = ask_ollama(observation, history)
                save_json(
                    f"step_{step_idx:02d}_decision.json",
                    {
                        "thought": decision.thought,
                        "expected_check": decision.expected_check,
                        "action": asdict(decision.action),
                        "bug_suspected": decision.bug_suspected,
                        "bug": decision.bug,
                    },
                )

                print(f"\nSTEP {step_idx}")
                print(f"THOUGHT: {decision.thought}")
                print(f"CHECK:   {decision.expected_check}")
                print(f"ACTION:  {decision.action.type}")

                action_result = execute_action(page, decision.action)
                action_history.append(action_result)
                print(f"RESULT:  {action_result}")

                report = build_bug_report(decision, action_history)
                if report:
                    already_exists = any(existing.title == report.title for existing in bug_reports)
                    if not already_exists:
                        bug_reports.append(report)
                        filename_base = f"bug_{len(bug_reports):02d}"
                        save_json(filename_base + ".json", asdict(report))
                        save_markdown(filename_base + ".md", bug_report_to_markdown(report))
                        print(f"BUG:     {report.title}")

                history.append(
                    {
                        "step": step_idx,
                        "observation": observation,
                        "decision": {
                            "thought": decision.thought,
                            "expected_check": decision.expected_check,
                            "action": asdict(decision.action),
                            "bug_suspected": decision.bug_suspected,
                            "bug": decision.bug,
                        },
                        "action_result": action_result,
                    }
                )
                save_json("run_history.json", history)

                if decision.action.type == "DONE":
                    print("Agent finished the test flow.")
                    break

            except (RuntimeError, ValueError, PlaywrightTimeoutError, ValidationError, json.JSONDecodeError) as e:
                print(f"\nSTEP {step_idx} FAILED: {e}")
                save_json(f"step_{step_idx:02d}_error.json", {"step": step_idx, "error": str(e)})
                break

        browser.close()

    print("\nFinished.")
    print(f"Saved logs to ./{LOG_DIR}")
    print(f"Bug reports drafted: {len(bug_reports)}")


if __name__ == "__main__":
    main()
