"""
The parts of the Whisper node that are plain Python: the language list the
card offers and the text the timings come back as.

Kept apart from anymatix_whisper.py so the tests read them without torch,
transformers or ComfyUI on the path.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# Whisper large-v3's own language table (openai/whisper tokenizer.py LANGUAGES,
# plus Cantonese, which v3 added). The card shows "Name (code)"; the node splits
# the code back out, the same contract as the Chatterbox language dropdown.
WHISPER_LANGUAGES = (
    ("en", "English"), ("zh", "Chinese"), ("de", "German"), ("es", "Spanish"),
    ("ru", "Russian"), ("ko", "Korean"), ("fr", "French"), ("ja", "Japanese"),
    ("pt", "Portuguese"), ("tr", "Turkish"), ("pl", "Polish"), ("ca", "Catalan"),
    ("nl", "Dutch"), ("ar", "Arabic"), ("sv", "Swedish"), ("it", "Italian"),
    ("id", "Indonesian"), ("hi", "Hindi"), ("fi", "Finnish"), ("vi", "Vietnamese"),
    ("he", "Hebrew"), ("uk", "Ukrainian"), ("el", "Greek"), ("ms", "Malay"),
    ("cs", "Czech"), ("ro", "Romanian"), ("da", "Danish"), ("hu", "Hungarian"),
    ("ta", "Tamil"), ("no", "Norwegian"), ("th", "Thai"), ("ur", "Urdu"),
    ("hr", "Croatian"), ("bg", "Bulgarian"), ("lt", "Lithuanian"), ("la", "Latin"),
    ("mi", "Maori"), ("ml", "Malayalam"), ("cy", "Welsh"), ("sk", "Slovak"),
    ("te", "Telugu"), ("fa", "Persian"), ("lv", "Latvian"), ("bn", "Bengali"),
    ("sr", "Serbian"), ("az", "Azerbaijani"), ("sl", "Slovenian"), ("kn", "Kannada"),
    ("et", "Estonian"), ("mk", "Macedonian"), ("br", "Breton"), ("eu", "Basque"),
    ("is", "Icelandic"), ("hy", "Armenian"), ("ne", "Nepali"), ("mn", "Mongolian"),
    ("bs", "Bosnian"), ("kk", "Kazakh"), ("sq", "Albanian"), ("sw", "Swahili"),
    ("gl", "Galician"), ("mr", "Marathi"), ("pa", "Punjabi"), ("si", "Sinhala"),
    ("km", "Khmer"), ("sn", "Shona"), ("yo", "Yoruba"), ("so", "Somali"),
    ("af", "Afrikaans"), ("oc", "Occitan"), ("ka", "Georgian"), ("be", "Belarusian"),
    ("tg", "Tajik"), ("sd", "Sindhi"), ("gu", "Gujarati"), ("am", "Amharic"),
    ("yi", "Yiddish"), ("lo", "Lao"), ("uz", "Uzbek"), ("fo", "Faroese"),
    ("ht", "Haitian Creole"), ("ps", "Pashto"), ("tk", "Turkmen"), ("nn", "Nynorsk"),
    ("mt", "Maltese"), ("sa", "Sanskrit"), ("lb", "Luxembourgish"), ("my", "Myanmar"),
    ("bo", "Tibetan"), ("tl", "Tagalog"), ("mg", "Malagasy"), ("as", "Assamese"),
    ("tt", "Tatar"), ("haw", "Hawaiian"), ("ln", "Lingala"), ("ha", "Hausa"),
    ("ba", "Bashkir"), ("jw", "Javanese"), ("su", "Sundanese"), ("yue", "Cantonese"),
)

AUTO_LANGUAGE = "Detect automatically"

LANGUAGE_OPTIONS: List[str] = [AUTO_LANGUAGE] + sorted(
    f"{name} ({code})" for code, name in WHISPER_LANGUAGES
)


def language_code(option: str) -> Optional[str]:
    """`Italian (it)` -> `it`; the automatic option -> None. Anything else is refused."""
    if option == AUTO_LANGUAGE:
        return None
    if option.endswith(")") and " (" in option:
        code = option[option.rindex(" (") + 2 : -1]
        if any(code == c for c, _ in WHISPER_LANGUAGES):
            return code
    raise ValueError(f"Whisper has no language {option!r}. Pick one of the listed options.")


def format_timings(chunks: Iterable[dict]) -> str:
    """One line per chunk, `start-end text`, seconds to two decimals.

    A chunk is what the transformers ASR pipeline returns with
    return_timestamps: {"text": str, "timestamp": (start, end)}. The end of the
    LAST chunk can be None when the audio stops mid-word; it is printed as the
    start, so every line still parses as two numbers.
    """
    lines = []
    for chunk in chunks:
        start, end = chunk["timestamp"]
        start = 0.0 if start is None else float(start)
        end = start if end is None else float(end)
        text = str(chunk["text"]).strip()
        if text:
            lines.append(f"{start:.2f}-{end:.2f} {text}")
    return "\n".join(lines)
