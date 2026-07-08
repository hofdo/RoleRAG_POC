import { expect, test } from "@playwright/test";

// Deterministic frontend<->backend contract test (#60). Runs the real Angular SPA
// against the fake-provider ASGI app (app.diagnostics.contract_app) over real HTTP
// -- no model, no Qdrant. Point at it with PLAYWRIGHT_BASE_URL, e.g.
//   PLAYWRIGHT_BASE_URL=http://127.0.0.1:18099 npm run test:e2e-contract
// Unlike spa-play.spec.mjs (which needs a live model), this asserts the fake
// provider's fixed line renders verbatim, so it is safe to gate CI on.

test("SPA creates a session and renders a fake-provider turn", async ({ page }) => {
  await page.goto("/app/");

  // Shell + setup picker present, catalog loaded from real data/.
  await expect(page.getByText("RoleRAG").first()).toBeVisible();
  await expect(page.getByRole("combobox", { name: "World", exact: true })).toBeEnabled();

  await page.getByLabel("Player name").fill("Contract");
  await page.getByRole("button", { name: "Start session" }).click();
  await expect(page.getByText(/Session active/)).toBeVisible();

  await page.getByPlaceholder("Type your turn…").fill("I promise to return before dawn.");
  await page.getByRole("button", { name: "Send" }).click();

  // The fake provider (SmokeTestProvider) returns a fixed actor line; asserting it
  // renders verbatim is the contract: backend response shape -> rendered transcript.
  const assistant = page.locator(".transcript .row.assistant .text, .log .row.assistant .text");
  await expect(assistant.first()).toContainText("Return before dawn", { timeout: 15_000 });
});
