import { expect, test } from "@playwright/test";

// Live e2e for the Angular SPA at /app. Needs the full stack running (make dev) and a model
// server — point at it with PLAYWRIGHT_BASE_URL, e.g.
//   PLAYWRIGHT_BASE_URL=http://127.0.0.1:8000 npm run test:e2e-spa

test("SPA creates a session, runs a local turn, and inspects it", async ({ page }) => {
  await page.goto("/app/");

  // Shell + setup picker present, catalog loaded.
  await expect(page.getByText("RoleRAG").first()).toBeVisible();
  await expect(page.getByLabel("World")).toBeEnabled();

  await page.getByLabel("Player name").fill("Playwright");
  await page.getByRole("button", { name: "Start session" }).click();

  // Setup picker collapses to the active-session line.
  await expect(page.getByText(/Session active/)).toBeVisible();

  await page
    .getByPlaceholder("Type your turn…")
    .fill("I ask what the Rose Gallery remembers about the regent.");
  await page.getByRole("button", { name: "Send" }).click();

  // Buffered turn can take minutes on a local model.
  const assistant = page.locator(".transcript .row.assistant .text, .log .row.assistant .text");
  await expect(assistant.first()).not.toHaveText("", { timeout: 150_000 });

  // Inspector lists the session's turns and drills into retrieval diagnostics.
  await page.getByRole("link", { name: "Inspector" }).click();
  await expect(page.getByRole("heading", { name: "RAG Inspector" })).toBeVisible();
  await page.getByLabel("Session").selectOption({ index: 1 });
  await page.locator(".timeline button").first().click();
  await expect(page.getByRole("heading", { name: /Turn #/ })).toBeVisible();
});
