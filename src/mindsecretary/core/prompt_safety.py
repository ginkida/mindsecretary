from __future__ import annotations


def sanitize_for_context(text: str, max_len: int = 500) -> str:
    """Sanitize user-origin text before injecting into a system prompt.

    Caps length and neutralizes instruction-like patterns. Defense in depth —
    not sufficient on its own; always pair with a role-lock stanza in the
    system prompt.
    """
    if not text:
        return ""
    text = text[:max_len]
    for prefix in (
        # English
        "## ", "# ", "System:", "SYSTEM:", "Instructions:",
        "You are", "You must", "Ignore previous", "Forget",
        "Assistant:", "Human:", "<system>", "</system>",
        # Russian
        "Системная инструкция", "Инструкция:", "Ты должен",
        "Забудь предыдущие", "Забудь всё", "Игнорируй",
        "Новая роль", "Новая задача", "Ты теперь",
    ):
        text = text.replace(prefix, f"[{prefix.strip()}]")
    return text
