# utils/text_wrapper.py
from __future__ import annotations


def wrap_two_lines(text: str, max_chars: int = 28) -> str:
    """
    Wrap simple: divide en máx 2 líneas (por palabras) si supera max_chars.
    Ideal para monospace en headers, títulos y leyendas.
    """
    s = (text or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s

    words = s.split()
    line1, line2 = "", ""

    for i, w in enumerate(words):
        test = (line1 + " " + w).strip()
        if len(test) <= max_chars:
            line1 = test
        else:
            # todo lo restante va a la segunda línea
            line2 = " ".join(words[i:]).strip()
            break

    if not line2:
        return line1

    # si la 2da línea quedó larguísima, la cortamos suave
    if len(line2) > max_chars:
        line2 = line2[: max_chars - 1].rstrip() + "…"

    return f"{line1}\n{line2}"
