"""
Simulation Output Post-Processor

Translates generated agent profile text fields (bio, persona, etc.) to the
target locale AFTER they have been produced by LLM or rule-based generation.

Design principles:
- Never modifies system prompts or instruction prompts
- Only processes the final text output of profile fields
- Degrades gracefully: if translation fails, original text is preserved
- Skips translation when source and target language already match
"""

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import Config
from ..utils.locale import get_language_instruction, get_locale
from ..utils.logger import get_logger

logger = get_logger("mirofish.postprocessor")

# Fields that contain free-form text and should be translated
_TEXT_FIELDS = ["bio", "persona", "profession"]
_LIST_FIELDS = ["interested_topics"]

# Heuristic: a string is "Chinese" if >10 % of its characters are CJK
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]")


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk_chars = len(_CJK_RE.findall(text))
    return cjk_chars / len(text)


def _needs_translation(text: str, target_locale: str) -> bool:
    """
    Decide whether *text* needs to be translated for *target_locale*.

    Logic:
    - If target is Chinese ('zh') and text has <10 % CJK → translate to Chinese
    - If target is non-Chinese and text has ≥10 % CJK  → translate away from Chinese
    - Otherwise assume text is already in an acceptable language
    """
    if not text or not text.strip():
        return False
    ratio = _cjk_ratio(text)
    if target_locale == "zh":
        return ratio < 0.10
    else:
        return ratio >= 0.10


class SimulationOutputPostProcessor:
    """
    Post-processes agent profile dicts to ensure text fields are in the
    correct language for the current locale.

    Usage::

        processor = SimulationOutputPostProcessor()
        profiles = processor.translate_profiles(profiles, locale="en")
    """

    _SYSTEM_PROMPT = (
        "You are a professional translator.  "
        "You will receive a JSON object whose string values are social-media "
        "persona descriptions.  Translate ONLY the string values into the "
        "requested language.  Keep all JSON keys unchanged.  Do NOT add, "
        "remove, or reorder keys.  Return valid JSON only, no markdown fences."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_profiles(
        self,
        profiles: List[Dict[str, Any]],
        locale: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Translate text fields of every profile dict in *profiles*.

        Args:
            profiles: List of profile dicts (as produced by OasisProfileGenerator).
            locale:   Target locale code (e.g. 'en', 'zh', 'vi').
                      Defaults to the current request/thread locale.

        Returns:
            A new list of profile dicts with translated text fields.
            Profiles that do not need translation are returned as-is.
        """
        target_locale = locale or get_locale()

        # Collect indices of profiles that actually need work
        to_translate: List[int] = []
        for idx, profile in enumerate(profiles):
            if self._profile_needs_translation(profile, target_locale):
                to_translate.append(idx)

        if not to_translate:
            logger.debug(
                "Post-processor: no profiles need translation "
                f"(locale={target_locale}, total={len(profiles)})"
            )
            return profiles

        logger.info(
            f"Post-processor: translating {len(to_translate)}/{len(profiles)} "
            f"profiles → locale={target_locale}"
        )

        result = list(profiles)  # shallow copy; we replace modified entries
        for idx in to_translate:
            try:
                result[idx] = self._translate_profile(profiles[idx], target_locale)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Post-processor: translation failed for profile index {idx}: {exc}. "
                    "Keeping original."
                )
                # Original profile is already in result[idx], nothing to do

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("LLM_API_KEY is not configured")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _profile_needs_translation(
        self, profile: Dict[str, Any], target_locale: str
    ) -> bool:
        """Return True if any translatable field needs translation."""
        for field in _TEXT_FIELDS:
            value = profile.get(field, "")
            if isinstance(value, str) and _needs_translation(value, target_locale):
                return True
        for field in _LIST_FIELDS:
            items = profile.get(field, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and _needs_translation(item, target_locale):
                        return True
        return False

    def _translate_profile(
        self, profile: Dict[str, Any], target_locale: str
    ) -> Dict[str, Any]:
        """
        Send relevant text fields to the LLM for translation and return a
        new profile dict with translated values merged back in.
        """
        # Build a minimal dict of only the fields that need translation
        payload: Dict[str, Any] = {}
        for field in _TEXT_FIELDS:
            value = profile.get(field, "")
            if isinstance(value, str) and _needs_translation(value, target_locale):
                payload[field] = value
        for field in _LIST_FIELDS:
            items = profile.get(field, [])
            if isinstance(items, list) and any(
                isinstance(i, str) and _needs_translation(i, target_locale)
                for i in items
            ):
                payload[field] = items

        if not payload:
            return profile

        lang_instruction = get_language_instruction()
        user_message = (
            f"{lang_instruction}\n\n"
            f"Translate the following JSON values:\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # low temperature for faithful translation
        )

        raw = response.choices[0].message.content
        translated: Dict[str, Any] = json.loads(raw)

        # Merge translated fields into a copy of the original profile
        updated = dict(profile)
        for key, value in translated.items():
            if key in payload:  # only accept keys we sent
                updated[key] = value

        return updated
