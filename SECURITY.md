# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately via [GitHub Security Advisories](https://github.com/ginkida/mindsecretary/security/advisories/new).

Do not open a public issue for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Design

- **Single-user auth**: Every Telegram handler checks `TELEGRAM_USER_ID`
- **SQL injection protection**: Parameterized queries, column name whitelists
- **Prompt injection mitigation**: Input sanitization before system prompt injection
- **Secrets**: `.env` file, never committed to git
- **DoS limits**: Voice 25MB/10min, photo 10MB, text 10K chars, processing timeout 90s
- **Docker**: Non-root user, read-only config volume
