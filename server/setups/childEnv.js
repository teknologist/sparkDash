/**
 * Environment for model start/stop scripts: the dashboard's own environment
 * MINUS its configuration and secrets.
 *
 * Model launchers read common names like PORT from the environment (GLM's
 * start.sh: "PORT=5600 from the environment overrides .env value 8888"), so
 * inheriting the dashboard's PORT would bind a model to the dashboard's port.
 * Tokens and the secrets key must never reach third-party scripts either.
 */
const DROP = new Set([
  "PORT",
  "BIND_HOST",
  "LLM_PORT",
  "COMFY_PORT",
  "SPARKDASH_TOKEN",
  "DASHBOARD_TOKEN",
  "SPARKDASH_SECRETS_KEY",
  "SPARKDASH_ALLOWED_HOSTS",
  "SPARKDASH_ALLOW_OPEN_REMOTE",
]);
/** Dashboard-internal file locations (SPARKS_JSON_PATH, …) mean nothing to a model. */
const DROP_RE = /^(SPARKS?_|SECRETS_|SETTINGS_|LLM_|FLEET_|EVENTS_|GPU_|BENCH_|PREFILL_|QUALITY_|TOOL_EVAL_|SHOWCASE_|POLL_INTERVAL_|MODEL_SETUPS_|MODELS_|COMPOSITIONS_|SETUP_LOGS_|LLAMA_SWAP_|HOST_(PROC|SYS|ROOT)_)/;

export function childEnv(extra = {}, base = process.env) {
  const env = {};
  for (const [k, v] of Object.entries(base)) {
    if (!DROP.has(k) && !DROP_RE.test(k)) env[k] = v;
  }
  return { ...env, ...extra };
}
