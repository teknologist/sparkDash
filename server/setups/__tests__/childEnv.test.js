import { test } from "node:test";
import assert from "node:assert/strict";
import { childEnv } from "../childEnv.js";

// GLM's start.sh reported "PORT=5600 from the environment overrides .env value
// 8888": the dashboard's own PORT leaked into a model script and would have
// bound the model to the dashboard's port. Secrets must not leak either.
test("model scripts do not inherit the dashboard's port, tokens or config paths", () => {
  const env = childEnv(
    { GPU_UTIL: "0.300", ENABLE_SPEC: "1" },
    { PATH: "/usr/bin", HOME: "/home/eric", PORT: "5600", BIND_HOST: "0.0.0.0", SPARKDASH_TOKEN: "t", SPARKDASH_SECRETS_KEY: "k", SPARKS_JSON_PATH: "/x", LLAMA_SWAP_URL: "u" }
  );
  assert.deepEqual(env, { PATH: "/usr/bin", HOME: "/home/eric", GPU_UTIL: "0.300", ENABLE_SPEC: "1" });
});

test("a brick's own env still wins, even for a dropped name", () => {
  assert.equal(childEnv({ PORT: "8002" }, { PORT: "5600" }).PORT, "8002");
});
