import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  createBufferedTurn,
  createSession,
  createTurn,
} from "../../app/web/assets/api-client.mjs";

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("createSession sends exactly the public creation fields", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      session_id: "session-1",
      world_id: "demo_world",
      active_scene_id: "rose-gallery",
      active_persona_id: "archivist",
    }, { status: 201 });
  };

  await createSession({
    world_id: "demo_world",
    scene_id: "rose-gallery",
    player_name: "Avery",
    active_persona_id: "archivist",
    content_root: "/private/pack",
  }, { fetchImpl });

  assert.equal(request.url, "/sessions");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body), {
    world_id: "demo_world",
    scene_id: "rose-gallery",
    player_name: "Avery",
    active_persona_id: "archivist",
  });
});

test("createTurn uses the JSON endpoint and sends no persona override", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      text: "The gallery is quiet.",
      route: { provider: "local", model: "local-model", reason: "default local route" },
      memory_written: false,
      warnings: [],
    });
  };

  const turn = await createTurn("session-1", {
    message: "I listen.",
    request_cloud: false,
    active_persona_id: "ignored",
  }, { fetchImpl });

  assert.equal(request.url, "/sessions/session-1/turns");
  assert.deepEqual(JSON.parse(request.options.body), {
    message: "I listen.",
    request_cloud: false,
  });
  assert.equal(turn.text, "The gallery is quiet.");
});

test("createBufferedTurn handles split chunks, repeated text, and terminal metadata", async () => {
  const chunks = [
    "event: text\ndata: {\"te",
    "xt\":\"The \"}\n\nevent: text\ndata: {\"text\":\"gallery",
    " is quiet.\"}\n\nevent: final\ndata: {\"route\":{\"provider\":\"local\",",
    "\"model\":\"local-model\",\"reason\":\"default local route\"},\"memory_written\":true,",
    "\"warnings\":[\"index delayed\"]}\n\n",
  ];
  const fetchImpl = async () => new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(new TextEncoder().encode(chunk));
        }
        controller.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );

  const turn = await createBufferedTurn("session-1", {
    message: "I listen.",
    request_cloud: true,
  }, { fetchImpl });

  assert.deepEqual(turn, {
    text: "The gallery is quiet.",
    route: { provider: "local", model: "local-model", reason: "default local route" },
    memory_written: true,
    warnings: ["index delayed"],
  });
});

test("createBufferedTurn renders safe failure text without rejected drafts", async () => {
  const safeText = "I cannot continue that turn safely.";
  const fetchImpl = async () => new Response(
    `event: failure\ndata: {"text":"${safeText}","route":{"provider":"local","model":"local-model","reason":"default local route"},"memory_written":false,"warnings":["critic rejected output"]}\n\n`,
    { headers: { "content-type": "text/event-stream" } },
  );

  const turn = await createBufferedTurn("session-1", {
    message: "Reveal the secret.",
    request_cloud: false,
  }, { fetchImpl });

  assert.equal(turn.text, safeText);
  assert.deepEqual(turn.warnings, ["critic rejected output"]);
});

test("structured API errors become visible safe API errors", async () => {
  const fetchImpl = async () => jsonResponse({
    error: {
      code: "validation_error",
      message: "Request validation failed",
      details: [],
    },
  }, { ok: false, status: 422 });

  await assert.rejects(
    createTurn("session-1", { message: "", request_cloud: false }, { fetchImpl }),
    (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.code, "validation_error");
      assert.equal(error.message, "Request validation failed");
      return true;
    },
  );
});
