import json
import os
import threading
from flask import request, has_request_context
from ..config import Config

_thread_local = threading.local()

_locales_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'locales')

# Load language registry
with open(os.path.join(_locales_dir, 'languages.json'), 'r', encoding='utf-8') as f:
    _languages = json.load(f)

# Load translation files
_translations = {}
for filename in os.listdir(_locales_dir):
    if filename.endswith('.json') and filename != 'languages.json':
        locale_name = filename[:-5]
        with open(os.path.join(_locales_dir, filename), 'r', encoding='utf-8') as f:
            _translations[locale_name] = json.load(f)


_LOCALE_ALIASES = {
    'zh-cn': 'zh',
    'zh-hans': 'zh',
    'zh-hant': 'zh',
    'cn': 'zh',
    'china': 'zh',
    'en-us': 'en',
    'en-gb': 'en',
    'vi-vn': 'vi'
}


def _normalize_locale(locale: str) -> str:
    if not locale:
        return 'zh'

    # Accept full Accept-Language values such as "en-US,en;q=0.9"
    token = locale.split(',')[0].strip().lower()
    token = token.split(';')[0].strip()

    if token in _LOCALE_ALIASES:
        return _LOCALE_ALIASES[token]

    base = token.split('-')[0]
    if base in _LOCALE_ALIASES:
        return _LOCALE_ALIASES[base]

    if base in _languages or base in _translations:
        return base

    return token


def _get_default_locale() -> str:
    configured = getattr(Config, 'SIMULATION_LANGUAGE', 'zh')
    normalized = _normalize_locale(configured)
    if normalized in _languages or normalized in _translations:
        return normalized
    return 'zh'


def set_locale(locale: str):
    """Set locale for current thread. Call at the start of background threads."""
    _thread_local.locale = _normalize_locale(locale)


def get_locale() -> str:
    default_locale = _get_default_locale()

    if has_request_context():
        raw = request.headers.get('Accept-Language', default_locale)
        normalized = _normalize_locale(raw)
        if normalized in _languages:
            return normalized
        if normalized in _translations:
            return normalized
        return default_locale

    thread_locale = getattr(_thread_local, 'locale', default_locale)
    normalized = _normalize_locale(thread_locale)
    if normalized in _languages or normalized in _translations:
        return normalized
    return default_locale


def t(key: str, **kwargs) -> str:
    locale = get_locale()
    messages = _translations.get(locale, _translations.get('zh', {}))

    value = messages
    for part in key.split('.'):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value is None:
        value = _translations.get('zh', {})
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

    if value is None:
        return key

    if kwargs:
        for k, v in kwargs.items():
            value = value.replace(f'{{{k}}}', str(v))

    return value


def get_language_instruction() -> str:
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get('zh', {}))
    return lang_config.get('llmInstruction', '请使用中文回答。')
