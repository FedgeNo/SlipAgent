# `http_fetch`

## HTTP Request

Send an HTTP(S) request to a public URL with an optional method, headers, query parameters, and body. Use this for APIs and raw responses; use `fetch_page` for readable page text.

## Request Body

Supply `body` for literal UTF-8 text or `json_body` for a JSON object or array. Pass JSON without pre-serializing it. Use `body` with a JSON content-type header for scalar JSON values. Set headers as needed for authentication or content type. Requests can change remote state; use methods and credentials appropriate to the user's request.

## Response

Returns `url`, `method`, `status_code`, `headers` (name/value pairs preserving duplicates), `truncated`, and `body`. Undecodable bodies use `body_base64` instead. HTTP 4xx/5xx responses are marked as errors and retain their bodies and headers. Transport failures return `error`.

Redirects are returned without following them. Inspect the location and choose the next request explicitly. Downloads are limited to 32 MB of decoded response bytes; `truncated` reports omitted bytes. The timeout defaults to 120 seconds, with a ceiling of 600 seconds. Only public HTTP(S) destinations without URL credentials are accepted; use headers for authentication. TLS certificates are verified. This tool does not run a shell or require cURL.
