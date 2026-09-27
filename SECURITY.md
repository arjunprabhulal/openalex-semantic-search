# Security policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub:
**Security → Report a vulnerability** on this repository. Do not open a public issue,
pull request or discussion for a suspected vulnerability.

Include what you found, how to reproduce it, and the impact you expect. Reports are
handled on a best-effort basis; this is a volunteer-maintained project and no response
time is promised.

## Supported versions

Only the latest release receives security fixes.

## Scope

In scope: the code in this repository, including the HTTP API, authentication, filter
rate limiting, and the ingestion and assembly pipeline.

Out of scope: third-party deployments of this software, the OpenAlex data itself, and
issues that need a valid API key plus access to the server's filesystem.

## Running it safely

- Bind the server to `127.0.0.1` and put a TLS reverse proxy in front.
- Keep API keys in environment files or a secret manager. Never commit them and never
  put them in browser JavaScript.
- Leave `OPENALEX_SEARCH_EXPOSE_OPENAPI` and `OPENALEX_SEARCH_UI` off in production
  unless you need them.
- Run the service as an unprivileged user with read-only access to the index.
