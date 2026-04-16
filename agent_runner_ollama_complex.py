import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

from ollama import chat
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, ValidationError


APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000/toy_app_complex.html")
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
LOG_DIR = os.getenv("LOG_DIR", "logs_ollama_complex")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")

APP_DESCRIPTION = (
    "This is a richer ecommerce QA test app with catalog search/filtering, a product page, a size-guide modal, "
    "cart summary, coupon logic, delayed shipping estimate, checkout validation, a settings page, and a success page."
)

TEST_GOAL = (
    "Test a realistic checkout flow. Search for jacket, filter outerwear, open the Winter Jacket page, open and close the size guide, "
    "select size M, set quantity 2, add the item to cart, go to cart, apply coupon SUMMER20, estimate shipping and wait for the result, "
    "go to checkout, fill valid contact info but invalid ZIP 12, place the order, and detect visible failures. "
    "Suspect a pricing bug if the coupon discount is not $20 for one $100 jacket. Suspect a validation bug if checkout succeeds with ZIP 12."
)

ALLOWED_ACTIONS = [
    "CLICK_TEXT",
    "TYPE_TEXT",
    "SELECT_OPTION",
    "WAIT_FOR_TEXT",
    "DONE",
]

SYSTEM_PROMPT = f"""
You are a cautious QA testing agent.

You receive:
- an app description
- a test goal
- the current observation of the page
- the history of previous steps

Your job:
1. Choose the next single action that best advances the test goal.
2. Detect suspicious visible behavior and describe it as a bug when warranted.
3. Return strict JSON only.

Rules:
- Do not invent UI elements.
- Take one small action at a time.
- Use only these action types: {', '.join(ALLOWED_ACTIONS)}.
- If the flow is complete, return DONE.
- If you suspect a bug, include a concise bug object.
- Prefer deterministic progress over exploration.

Action rules:
- CLICK_TEXT: include target_text
- TYPE_TEXT: include field_text and input_text
- SELECT_OPTION: include field_text and option_text
- WAIT_FOR_TEXT: include target_text and wait_ms
- DONE: leave optional fields null
""".strip()


class ActionModel(BaseModel):
    type: Literal["CLICK_TEXT", "TYPE_TEXT", "SELECT_OPTION", "WAIT_FOR_TEXT", "DONE"]
    target_text: Optional[str] = None
    field_text: Optional[str] = None
    input_text: Optional[str] = None
    option_text: Optional[str] = None
    wait_ms: Optional[int] = None


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
    option_text: Optional[str] = None
    wait_ms: Optional[int] = None


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
    source: str = "ollama"


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


def compact_lines(text: str, max_chars: int = 1800) -> str:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned[:max_chars]


def build_observation(page) -> Dict[str, Any]:
    title = page.title()
    url = page.url

    buttons: List[str] = []
    for locator in [
        page.locator("button:visible"),
        page.locator("a:visible"),
        page.locator("[role='button']:visible"),
    ]:
        count = locator.count()
        for i in range(count):
            try:
                text = locator.nth(i).inner_text().strip()
                if text and text not in buttons:
                    buttons.append(text)
            except Exception:
                pass

    inputs: List[Dict[str, str]] = []
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

    selects: List[Dict[str, str]] = []
    select_locator = page.locator("select:visible")
    for i in range(select_locator.count()):
        try:
            element = select_locator.nth(i)
            name = element.get_attribute("name") or f"select_{i}"
            value = element.input_value()
            options = [opt.strip() for opt in element.locator("option").all_inner_texts() if opt.strip()]
            selects.append({"label": name, "value": value, "options": options})
        except Exception:
            pass

    body_text = page.locator("body").inner_text(timeout=3000)

    return {
        "title": title,
        "url": url,
        "buttons": buttons,
        "inputs": inputs,
        "selects": selects,
        "body_text": compact_lines(body_text),
    }


def sanitize_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text


def get_input_value(observation: Dict[str, Any], label: str) -> str:
    for item in observation.get("inputs", []):
        if item["label"].strip().lower() == label.strip().lower():
            return item["value"]
    return ""


def get_select_value(observation: Dict[str, Any], label: str) -> str:
    for item in observation.get("selects", []):
        if item["label"].strip().lower() == label.strip().lower():
            return item["value"]
    return ""


def heuristic_decision(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> AgentDecision:
    buttons = observation.get("buttons", [])
    text = observation.get("body_text", "")
    # history_text = json.dumps(history)

    # def seen(fragment: str) -> bool:
    #     return fragment in history_text
    def seen(fragment: str) -> bool:
        return any(fragment in str(step.get("action_result", "")) for step in history)

    def seen_bug(title: str) -> bool:
        for step in history:
            decision = step.get("decision", {}) or {}
            bug = decision.get("bug")
            if isinstance(bug, dict) and bug.get("title") == title:
                return True
        return False

    if "Apply Filters" in buttons and not seen('Typed into "Search products"'):
        return AgentDecision(
            thought="Start by narrowing the catalog to the target product.",
            expected_check="The search field should contain jacket.",
            action=Action(type="TYPE_TEXT", field_text="Search products", input_text="jacket"),
            bug_suspected=False,
            source="fallback",
        )

    if "Apply Filters" in buttons and get_select_value(observation, "Category") != "Outerwear":
        return AgentDecision(
            thought="Set the category filter to Outerwear before applying filters.",
            expected_check="Category should be set to Outerwear.",
            action=Action(type="SELECT_OPTION", field_text="Category", option_text="Outerwear"),
            bug_suspected=False,
            source="fallback",
        )

    if "Apply Filters" in buttons and not seen("Clicked 'Apply Filters'."):
        return AgentDecision(
            thought="Apply the current catalog filters.",
            expected_check="Winter Jacket should remain visible.",
            action=Action(type="CLICK_TEXT", target_text="Apply Filters"),
            bug_suspected=False,
            source="fallback",
        )

    if "View Product" in buttons and "Winter Jacket" in text and not seen("Clicked 'View Product'."):
        return AgentDecision(
            thought="Open the target product page.",
            expected_check="The product page should load for Winter Jacket.",
            action=Action(type="CLICK_TEXT", target_text="View Product"),
            bug_suspected=False,
            source="fallback",
        )

    if "Open Size Guide" in buttons and not seen("Clicked 'Open Size Guide'."):
        return AgentDecision(
            thought="Exercise the product modal once before adding to cart.",
            expected_check="The size guide modal should appear.",
            action=Action(type="CLICK_TEXT", target_text="Open Size Guide"),
            bug_suspected=False,
            source="fallback",
        )

    if "Close Size Guide" in buttons and not seen("Clicked 'Close Size Guide'."):
        return AgentDecision(
            thought="Close the size guide modal and continue the purchase flow.",
            expected_check="The product page should be visible again.",
            action=Action(type="CLICK_TEXT", target_text="Close Size Guide"),
            bug_suspected=False,
            source="fallback",
        )

    if "Add to Cart" in buttons and get_select_value(observation, "Size") != "M":
        return AgentDecision(
            thought="Select size M before adding the product to the cart.",
            expected_check="The size selector should show M.",
            action=Action(type="SELECT_OPTION", field_text="Size", option_text="M"),
            bug_suspected=False,
            source="fallback",
        )

    if "Add to Cart" in buttons and get_input_value(observation, "Quantity") != "1":
        return AgentDecision(
            thought="Use the default quantity of 1 so the expected coupon math stays simple.",
            expected_check="Quantity should be 1.",
            action=Action(type="TYPE_TEXT", field_text="Quantity", input_text="1"),
            bug_suspected=False,
            source="fallback",
        )

    if "Add to Cart" in buttons and not seen("Clicked 'Add to Cart'."):
        return AgentDecision(
            thought="Add the configured product to the cart.",
            expected_check="The product page should confirm that the item was added.",
            action=Action(type="CLICK_TEXT", target_text="Add to Cart"),
            bug_suspected=False,
            source="fallback",
        )

    if "Cart (1)" in buttons and not seen("Clicked 'Cart (1)'."):
        return AgentDecision(
            thought="Go to the cart after adding the item.",
            expected_check="The cart screen should appear.",
            action=Action(type="CLICK_TEXT", target_text="Cart (1)"),
            bug_suspected=False,
            source="fallback",
        )

    if "Apply Coupon" in buttons and get_input_value(observation, "Coupon") != "SUMMER20":
        return AgentDecision(
            thought="Enter the coupon code before testing the pricing logic.",
            expected_check="The coupon field should contain SUMMER20.",
            action=Action(type="TYPE_TEXT", field_text="Coupon", input_text="SUMMER20"),
            bug_suspected=False,
            source="fallback",
        )

    if "Apply Coupon" in buttons and not seen("Clicked 'Apply Coupon'."):
        return AgentDecision(
            thought="Apply the coupon and inspect the resulting summary.",
            expected_check="The discount should become $20 and the total should update accordingly.",
            action=Action(type="CLICK_TEXT", target_text="Apply Coupon"),
            bug_suspected=False,
            source="fallback",
        )

    # if "Discount" in text and "-$10" in text and not seen("Coupon applies incorrect total"):
    #     return AgentDecision(
    #         thought="The discount amount is visibly incorrect after coupon application.",
    #         expected_check="Discount should be -$20 for one $100 jacket with SUMMER20.",
    #         action=Action(type="CLICK_TEXT", target_text="Estimate Shipping"),
    #         bug_suspected=True,
    #         bug={
    #             "title": "Coupon applies incorrect discount",
    #             "severity": "Medium",
    #             "expected": "SUMMER20 should reduce a $100 subtotal by $20.",
    #             "actual": "The order summary shows Discount -$10 after applying SUMMER20.",
    #             "reason": "The pricing logic appears to apply the wrong discount amount.",
    #         },
    #         source="fallback",
    #     )


    if "Calculating shipping..." in text:
        return AgentDecision(
            thought="Wait for the asynchronous shipping estimate to complete.",
            expected_check="The page should show Shipping estimate ready.",
            action=Action(type="WAIT_FOR_TEXT", target_text="Shipping estimate ready.", wait_ms=2000),
            bug_suspected=False,
            source="fallback",
        )
    
    if "Discount" in text and "-$10" in text and not seen_bug("Coupon applies incorrect discount"):
        return AgentDecision(
            thought="The discount amount is visibly incorrect after coupon application.",
            expected_check="Discount should be -$20 for one $100 jacket with SUMMER20.",
            action=Action(type="CLICK_TEXT", target_text="Estimate Shipping"),
            bug_suspected=True,
            bug={
                "title": "Coupon applies incorrect discount",
                "severity": "Medium",
                "expected": "SUMMER20 should reduce a $100 subtotal by $20.",
                "actual": "The order summary shows Discount -$10 after applying SUMMER20.",
                "reason": "The pricing logic appears to apply the wrong discount amount.",
            },
            source="fallback",
        )

    if "Estimate Shipping" in buttons and not seen("Clicked 'Estimate Shipping'."):
        return AgentDecision(
            thought="Trigger the asynchronous shipping estimate.",
            expected_check="Shipping should update after the loading state finishes.",
            action=Action(type="CLICK_TEXT", target_text="Estimate Shipping"),
            bug_suspected=False,
            source="fallback",
        )

    if "Proceed to Checkout" in buttons and not seen("Clicked 'Proceed to Checkout'."):
        return AgentDecision(
            thought="Continue to the checkout form.",
            expected_check="The checkout page should appear.",
            action=Action(type="CLICK_TEXT", target_text="Proceed to Checkout"),
            bug_suspected=False,
            source="fallback",
        )

    if "Place Order" in buttons and get_input_value(observation, "Full Name") != "Taylor Rivera":
        return AgentDecision(
            thought="Fill the name field before submission.",
            expected_check="Full Name should be set.",
            action=Action(type="TYPE_TEXT", field_text="Full Name", input_text="Taylor Rivera"),
            bug_suspected=False,
            source="fallback",
        )

    if "Place Order" in buttons and get_input_value(observation, "Email") != "taylor@example.com":
        return AgentDecision(
            thought="Fill the email field before submission.",
            expected_check="Email should be set.",
            action=Action(type="TYPE_TEXT", field_text="Email", input_text="taylor@example.com"),
            bug_suspected=False,
            source="fallback",
        )

    if "Place Order" in buttons and get_input_value(observation, "Address") != "500 Forbes Ave":
        return AgentDecision(
            thought="Fill the address field before submission.",
            expected_check="Address should be set.",
            action=Action(type="TYPE_TEXT", field_text="Address", input_text="500 Forbes Ave"),
            bug_suspected=False,
            source="fallback",
        )

    if "Place Order" in buttons and get_input_value(observation, "ZIP Code") != "12":
        return AgentDecision(
            thought="Use an obviously invalid ZIP to test validation.",
            expected_check="ZIP Code should be set to 12 before submission.",
            action=Action(type="TYPE_TEXT", field_text="ZIP Code", input_text="12"),
            bug_suspected=False,
            source="fallback",
        )

    if "Place Order" in buttons and not seen("Clicked 'Place Order'."):
        return AgentDecision(
            thought="Submit the checkout form and check whether invalid ZIP is blocked.",
            expected_check="The app should show a validation error instead of reaching success.",
            action=Action(type="CLICK_TEXT", target_text="Place Order"),
            bug_suspected=False,
            source="fallback",
        )

    if "Processing order..." in text:
        return AgentDecision(
            thought="Wait for the order processing state to complete.",
            expected_check="The next visible state should reveal whether validation blocked submission.",
            action=Action(type="WAIT_FOR_TEXT", target_text="Order Confirmed", wait_ms=1500),
            bug_suspected=False,
            source="fallback",
        )

    if "Order Confirmed" in text and not seen("Checkout accepts invalid ZIP code"):
        return AgentDecision(
            thought="The app reached success even though ZIP 12 should have been rejected.",
            expected_check="The app should have remained on checkout with a validation error.",
            action=Action(type="DONE"),
            bug_suspected=True,
            bug={
                "title": "Checkout accepts invalid ZIP code",
                "severity": "High",
                "expected": "Checkout should block submission when ZIP 12 is entered.",
                "actual": "The app reached Order Confirmed after submitting ZIP 12.",
                "reason": "ZIP validation did not stop an obviously invalid submission.",
            },
            source="fallback",
        )

    return AgentDecision(
        thought="No useful next action remains for this bounded test flow.",
        expected_check="Testing is complete.",
        action=Action(type="DONE"),
        bug_suspected=False,
        source="fallback",
    )


def ask_ollama(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> AgentDecision:
    user_payload = {
        "app_description": APP_DESCRIPTION,
        "test_goal": TEST_GOAL,
        "observation": observation,
        "history": history[-6:],
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
            "num_predict": 120,
            "num_ctx": 1024,
        },
    )

    raw = sanitize_json_text(response.message.content)
    if not raw:
        raise ValueError("Ollama returned empty content")

    parsed = DecisionModel.model_validate_json(raw)
    return AgentDecision(
        thought=parsed.thought,
        expected_check=parsed.expected_check,
        action=Action(**parsed.action.model_dump()),
        bug_suspected=parsed.bug_suspected,
        bug=parsed.bug.model_dump() if parsed.bug else None,
        source="ollama",
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


def select_best_option(page, field_text: str, option_text: str) -> None:
    try:
        page.locator(f"select[name='{field_text}']").first.select_option(label=option_text, timeout=2000)
        return
    except Exception:
        pass

    select_locator = page.locator("select:visible")
    field_lower = field_text.lower()
    for i in range(select_locator.count()):
        try:
            element = select_locator.nth(i)
            name = (element.get_attribute("name") or "").lower()
            if field_lower in name:
                element.select_option(label=option_text, timeout=2000)
                return
        except Exception:
            pass

    raise RuntimeError(f"Could not find select for field: {field_text}")


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
        return f'Typed into "{action.field_text}".'

    if action.type == "SELECT_OPTION":
        if not action.field_text or not action.option_text:
            raise ValueError("SELECT_OPTION requires field_text and option_text")
        select_best_option(page, action.field_text, action.option_text)
        return f'Selected "{action.option_text}" for "{action.field_text}".'

    if action.type == "WAIT_FOR_TEXT":
        if not action.target_text:
            raise ValueError("WAIT_FOR_TEXT requires target_text")
        wait_ms = action.wait_ms or 1500
        try:
            page.get_by_text(action.target_text).first.wait_for(timeout=wait_ms)
        except Exception:
            page.wait_for_timeout(wait_ms)
        return f'Waited for "{action.target_text}".'

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
                page.wait_for_timeout(350)
                observation = build_observation(page)
                save_json(f"step_{step_idx:02d}_observation.json", observation)

                raw_response = None
                decision_source = "ollama"
                try:
                    decision = ask_ollama(observation, history)
                except Exception as llm_error:
                    decision = heuristic_decision(observation, history)
                    raw_response = {"ollama_error": str(llm_error), "fallback_used": True}
                    decision_source = "fallback"

                save_json(
                    f"step_{step_idx:02d}_decision.json",
                    {
                        "source": decision_source,
                        "thought": decision.thought,
                        "expected_check": decision.expected_check,
                        "action": asdict(decision.action),
                        "bug_suspected": decision.bug_suspected,
                        "bug": decision.bug,
                        "meta": raw_response,
                    },
                )

                print(f"\nSTEP {step_idx}")
                print(f"SOURCE:  {decision.source}")
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
                        "source": decision.source,
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
