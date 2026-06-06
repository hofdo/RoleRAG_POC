import { expect, test } from "@playwright/test";

test("live play UI creates a session and completes a local turn", async ({ page }) => {
  await page.goto("/play");

  await expect(page.getByRole("heading", { name: "RoleRAG Play" })).toBeVisible();
  await expect(page.locator("#catalog-world")).toBeEnabled();
  await expect(page.locator("#runtime-status-list")).toContainText("rolerag-poc");

  await page.locator("#player-name").fill("Playwright");
  await page.getByRole("button", { name: "Create session" }).click();

  await expect(page.locator("#play-panel")).toBeVisible();
  await expect(page.locator("#session-summary")).toContainText(
    "demo_world / rose-gallery / archivist",
  );

  await page
    .locator("#turn-message")
    .fill("I ask what the Rose Gallery remembers about the regent.");
  await page.getByRole("button", { name: "Send" }).click();

  const assistantMessages = page.locator("#transcript .transcript-entry.assistant p");
  await expect(assistantMessages).toHaveCount(1, { timeout: 150_000 });
  await expect(assistantMessages.first()).not.toHaveText("");
  await expect(page.locator("#debug-state")).toContainText("Route provider");
  await expect(page.locator("#debug-state")).toContainText("local");
});
