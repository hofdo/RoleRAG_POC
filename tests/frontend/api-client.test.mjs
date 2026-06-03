import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  createBufferedTurn,
  createSession,
  createTurn,
  getContentCatalog,
  getRecentSessions,
  getRuntimeStatus,
  getSession,
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

test("getContentCatalog uses the catalog endpoint with GET", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      worlds: [
        {
          id: "demo_world",
          name: "Winter Palace Intrigue",
          default_scene_id: "rose-gallery",
          scene_ids: ["rose-gallery"],
          persona_ids: ["archivist"],
        },
      ],
      scenes: [],
      personas: [],
    });
  };

  const catalog = await getContentCatalog({ fetchImpl });

  assert.equal(request.url, "/content/catalog");
  assert.equal(request.options.method, "GET");
  assert.equal("body" in request.options, false);
  assert.equal(catalog.worlds[0].id, "demo_world");
});

test("getRuntimeStatus uses the runtime endpoint with GET", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      app_name: "rolerag-poc",
      app_version: "0.1.0",
      environment: "local",
      cloud_mode: "ask",
      retrieval_configured: true,
      content_catalog_available: true,
      local_provider_configured: true,
      cloud_provider_configured: false,
    });
  };

  const status = await getRuntimeStatus({ fetchImpl });

  assert.equal(request.url, "/runtime/status");
  assert.equal(request.options.method, "GET");
  assert.equal("body" in request.options, false);
  assert.equal(status.app_name, "rolerag-poc");
  assert.equal(status.cloud_provider_configured, false);
});

test("getRecentSessions uses the sessions endpoint with GET and no body", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      sessions: [
        {
          session_id: "session-1",
          world_id: "demo_world",
          active_scene_id: "rose-gallery",
          active_persona_id: "archivist",
          player_name: "Avery",
          created_at: "2026-06-03T10:00:00Z",
          updated_at: "2026-06-03T10:05:00Z",
        },
      ],
    });
  };

  const response = await getRecentSessions({ fetchImpl });

  assert.equal(request.url, "/sessions");
  assert.equal(request.options.method, "GET");
  assert.equal("body" in request.options, false);
  assert.equal(response.sessions[0].session_id, "session-1");
});

test("structured catalog errors stay ApiError instances", async () => {
  const fetchImpl = async () => jsonResponse({
    error: {
      code: "invalid_content_catalog",
      message: "Configured content catalog is missing the worlds directory.",
      details: [],
    },
  }, { ok: false, status: 400 });

  await assert.rejects(
    getContentCatalog({ fetchImpl }),
    (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.code, "invalid_content_catalog");
      assert.equal(error.status, 400);
      return true;
    },
  );
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

test("getSession uses the encoded session lookup endpoint with GET", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return jsonResponse({
      session_id: "session/with spaces",
      world_id: "demo_world",
      active_scene_id: "rose-gallery",
      active_persona_id: "archivist",
      recent_turns: [],
    });
  };

  await getSession("session/with spaces", { fetchImpl });

  assert.equal(request.url, "/sessions/session%2Fwith%20spaces");
  assert.equal(request.options.method, "GET");
  assert.equal("body" in request.options, false);
});

test("getSession returns safe session fields and recent turns", async () => {
  const fetchImpl = async () => jsonResponse({
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
    recent_turns: [
      {
        turn_index: 1,
        user_message: "I listen.",
        assistant_message: "The gallery is quiet.",
        created_at: "2026-06-03T10:00:00Z",
      },
    ],
  });

  const session = await getSession("session-1", { fetchImpl });

  assert.deepEqual(session, {
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
    recent_turns: [
      {
        turn_index: 1,
        user_message: "I listen.",
        assistant_message: "The gallery is quiet.",
        created_at: "2026-06-03T10:00:00Z",
      },
    ],
  });
});

test("structured session lookup 404 becomes an ApiError", async () => {
  const fetchImpl = async () => jsonResponse({
    error: {
      code: "session_not_found",
      message: "Session not found: missing-session",
      details: [],
    },
  }, { ok: false, status: 404 });

  await assert.rejects(
    getSession("missing-session", { fetchImpl }),
    (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.code, "session_not_found");
      assert.equal(error.message, "Session not found: missing-session");
      assert.equal(error.status, 404);
      return true;
    },
  );
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
