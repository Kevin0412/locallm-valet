# Local JEV typed-decision models

Local JEV models expose `POST /v1/systemone`. Configure
`models.<name>.backend.api_protocol: systemone` to advertise Choice, Noul and
Score capabilities. Chat and streaming requests are rejected before loading
these models. Existing chat models keep the default `openai` protocol.

The gateway manages startup, health checks, request accounting, model switching,
VRAM gates and idle unload using the same GPU slot as the other local models.
It forwards named typed questions and responses, and records `input_tokens` /
`output_tokens` usage. Its authentication applies to the decision endpoint.

## Register the existing benchmark checkpoints

Prepare the open backends in the JEV benchmark repository first. The integration
requires its `scripts/serve_backend.py`, `configs/models.json`, backend lock,
clean pinned upstream checkouts, local checkpoints and isolated interpreters.
It does not download weights or call the official TypeSafe API.

```bash
python scripts/register_jev.py --jev-repository /home/test/jev_bench --dry-run
python scripts/register_jev.py --jev-repository /home/test/jev_bench
# Restart the gateway while it has no active requests/jobs.
```

The command adds all six local configurations, preserves other registry entries
and credentials, validates the candidate YAML, and saves a private backup under
ignored `data/config-backups/`. Existing conflicting JEV entries are rejected.
Use `--slot` for the existing GPU slot name; models on the same GPU must share it.
Per-model interpreters, model paths and checkpoint revisions come from the lock.
JEV runs with BF16 weights, offline Hub settings and a serialized scorer.

VRAM budgets are the observed full-suite peaks rounded up plus 1 GiB (6, 19, 10,
9, 10 and 11 GiB for the six configurations); the gateway adds its safety margin.
These are starting budgets for this measured setup, not universal memory bounds.
Unrelated GPU processes are counted by the gate and are not stopped automatically.

## Call the gateway

Use your **gateway** key, not the official JEV API key:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $LOCALLM_VALET_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"openjev_2b","state":"The customer was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","support":"Technical issues"}},"duplicate":{"type":"noul","instructions":"Was the customer charged twice?"}}}'
```

Score criteria are ordered descriptions (2–10 levels); Choice options have stable
string keys. There may be 1–32 named questions per request. SemIf supports up to
16 Choice options, mini-jev 26, and Open-Jev/simple-jev 255. SemIf and mini-jev
derive Score from categorical ordinal probabilities; the other implementations
use their upstream typed scorer. Responses retain the registry name in `model`.

## Status codes

`422` means the request itself is invalid (wrong question type, too many options,
too many questions, input above the checkpoint's token limit). The response
`error.message` names the reason and the same reason is logged by the model
service, which the gateway prefixes with `[backend]`. `500` means the scorer
answered in a shape that cannot be trusted (probabilities that do not sum to one,
a `choice` that is not the probability maximum, a `score` that disagrees with its
distribution); those are logged with a traceback. `400` comes from the gateway
itself when a typed model is called on a chat endpoint (or vice versa) and no
model is loaded.

Choice, Noul and Score usage is reported as `input_tokens`; the gateway records it
under the usual prompt-token column.

The original benchmark adapter, frozen input sets, evaluation results and
official API configuration remain separate from this local service.
