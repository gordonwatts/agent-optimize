# Optional OpenAI generation

Mock generation is the default, even when a credential is present. Install
`pip install -e '.[openai]'` and explicitly select a live provider:

```powershell
agent-optimize run --provider openai --model gpt-6.1-sol --max-output-tokens 4096 -- python analysis.py
```

Supported model IDs are `gpt-6.1-sol`, `gpt-6.1-terra`, and `gpt-6-luna`.
Availability depends on the API account. Live calls incur API charges. The key
lookup order is `--api-key`, `AGENT_OPTIMIZE_OAI_API_KEY` in the environment,
the project root's `.env`, then the user's `~/.env`. The project root is the
nearest ancestor containing `.git` or `pyproject.toml`, falling back to the
working directory. Credential files use
`AGENT_OPTIMIZE_OAI_API_KEY=...`; literal quoted values and `export` prefixes
are supported. There is no interpolation and no fallback to `OPENAI_API_KEY`.
Prefer the environment or an ignored `.env` over a command-line key because
shell history and operating-system command inspection can expose arguments.

The live implementation follows the [official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses):
the official Python SDK's `responses.parse` receives a Pydantic proposal schema.
It uses the official API endpoint, disables retries, sets `store=False`, and
bounds output tokens. The separate provider process has a fixed 180-second (three-minute) timeout, including SDK
startup. This is the only optimization timeout; local checking and timing have
no execution time limits. Refusals, incomplete output, malformed output, transport errors,
response limits, and timeouts produce distinct local statuses.

Requests include the frozen reference source (including its literal constants),
operation graph, type/schema contract, interface constraints, fixed synthetic
examples with zero-valued fields and scalar one, and prior candidate source plus
allowlisted evaluator feedback. Captured input buffers, scalar values, recording
identifiers, runtime exceptions, and private correctness cases are never included.
Candidate compiler diagnostics are included with paths and source locations,
without terminal styling, up to 24 KiB of JSON text; source validation messages
are also retained. The model therefore receives the actual code failure rather
than a generic correctness-failure label.
`--max-request-bytes` bounds the serialized local problem/history envelope,
including provider identity; it does not measure the full HTTP request, which
also contains fixed instructions, the schema, and SDK framing.
`--max-response-bytes` bounds the local parsed proposal after the SDK has read
the API response; it is not a streaming HTTP download limit. Token limits and the fixed model-request timeout
bound live generation. These local byte checks also apply to mock mode.

The credential goes to the provider process through a private pipe, never a
request file or child environment. Analysis, replay, correctness and timing
processes receive environments with credential variables removed. This is
process and data-flow isolation, not an operating-system filesystem sandbox:
analysis code runs with the user's normal file access. Treat analysis code as
trusted. SDK exceptions are reduced to fixed statuses; credentials and raw
HTTP diagnostics are never written into reports. Reports and accepted manifests
include provider/model/settings and allowlisted token usage when returned.
Each evaluation also records `provider_request_count`, counted at the HTTP
request hook with retries disabled. It is zero for mock/local preflight failure
and unknown (`null`) if the provider process is terminated before reporting.
Provider failures have a fixed `provider_error` code and never raw SDK text.

Offline fake-client tests run without credentials. The real-API test is skipped
unless `AGENT_OPTIMIZE_LIVE_TESTS=1` is explicitly set. It additionally needs a
credential and uses `AGENT_OPTIMIZE_LIVE_MODEL` (default `gpt-6-luna`).
