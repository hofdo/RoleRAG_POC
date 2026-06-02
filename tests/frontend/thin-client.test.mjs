import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ASSETS = [
  "api-client.mjs",
  "play-model.mjs",
  "play-ui.mjs",
];

test("frontend assets stay free of backend orchestration and secret-handling logic", async () => {
  const source = (
    await Promise.all(
      ASSETS.map((asset) => readFile(new URL(`../../app/web/assets/${asset}`, import.meta.url), "utf8")),
    )
  ).join("\n");

  for (const forbidden of [
    "content_root",
    "hidden_context",
    "prompt",
    "sqlite",
    "qdrant",
    "api_key",
    "provider_secret",
  ]) {
    assert.equal(source.toLowerCase().includes(forbidden), false, forbidden);
  }
});
