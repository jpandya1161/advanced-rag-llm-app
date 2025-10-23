"""Browser-level smoke test for the local RAG application."""

import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "browser"


def wait_until_enabled(page: Page, selector: str) -> None:
    page.locator(selector).wait_for(state="visible")
    page.wait_for_function(
        "selector => !document.querySelector(selector).disabled", arg=selector
    )


def assert_no_horizontal_overflow(page: Page) -> None:
    dimensions = page.evaluate(
        """() => ({
            viewport: document.documentElement.clientWidth,
            content: document.documentElement.scrollWidth
        })"""
    )
    assert dimensions["content"] <= dimensions["viewport"], dimensions


def run_workflow(page: Page) -> None:
    print("[browser] opening desktop app", flush=True)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator("#healthBadge").get_by_text("ok").wait_for()
    assert_no_horizontal_overflow(page)
    page.screenshot(path=ARTIFACTS / "desktop-initial.png", full_page=True)

    page.get_by_role("button", name="Clear Index").click()
    page.get_by_text("0 documents", exact=False).wait_for()
    print("[browser] index cleared", flush=True)

    page.locator("#documentInput").set_input_files(
        ROOT / "sample_docs" / "rag_background.txt"
    )
    page.get_by_role("button", name="Upload", exact=True).click()
    page.get_by_text("1 documents", exact=False).wait_for()
    page.get_by_text("rag_background", exact=False).first.wait_for()
    print("[browser] document upload verified", flush=True)

    wait_until_enabled(page, "#sampleButton")
    page.get_by_role("button", name="Load Samples").click()
    expected_documents = len(list((ROOT / "sample_docs").glob("*"))) + 1
    page.get_by_text(f"{expected_documents} documents", exact=False).wait_for()
    print("[browser] sample ingestion verified", flush=True)

    question = "What should the assistant do when retrieved evidence is missing?"
    page.get_by_placeholder("Ask: What should the assistant do when evidence is missing?").fill(
        question
    )
    page.get_by_role("button", name="Ask", exact=True).click()
    page.get_by_text(question, exact=True).wait_for()
    page.locator(".message.assistant .citation-chip").last.wait_for()
    page.locator(".source-item").first.wait_for()
    print("[browser] grounded answer and sources verified", flush=True)
    assert "ms" in page.locator("#latencyBadge").inner_text()
    page.screenshot(path=ARTIFACTS / "desktop-answer.png", full_page=True)

    wait_until_enabled(page, "#evalButton")
    page.get_by_role("button", name="Run Sample Eval").click()
    page.wait_for_function(
        "() => document.querySelector('#evalOutput').textContent.includes('average_fact_recall')"
    )
    assert "citation_or_refusal_rate" in page.locator("#evalOutput").inner_text()
    print("[browser] evaluation verified", flush=True)


def verify_mobile(page: Page) -> None:
    print("[browser] checking mobile layout", flush=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator("#healthBadge").get_by_text("ok").wait_for()
    assert_no_horizontal_overflow(page)
    assert page.get_by_role("heading", name="Knowledge Assistant").is_visible()
    assert page.get_by_role("button", name="Ask", exact=True).is_visible()
    page.screenshot(path=ARTIFACTS / "mobile.png", full_page=True)
    print("[browser] mobile layout verified", flush=True)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    console_errors = []
    page_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(15_000)
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        run_workflow(page)
        verify_mobile(page)
        browser.close()

    assert not page_errors, page_errors
    assert not console_errors, console_errors
    print("Browser smoke test passed: upload, retrieval, evaluation, and mobile layout")


if __name__ == "__main__":
    main()
