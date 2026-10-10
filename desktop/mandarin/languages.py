"""Explicit recognition languages and offline Chinese script conversion.

Chinese writing systems share the same Mandarin recognizer. OpenCC is loaded
only when conversion is requested; importing this module opens no model/device.
"""
from __future__ import annotations

from functools import lru_cache


LANGUAGE_CHOICES = (
    ('zh-Hans', '简体中文'),
    ('zh-Hant', '繁體中文'),
    ('en', 'English'),
)
DEFAULT_LANGUAGE = 'zh-Hans'


def validate_language(language: str) -> str:
    if not isinstance(language, str) or language not in tuple(item[0] for item in LANGUAGE_CHOICES):
        raise ValueError('不支持的语言，请选择简体中文、繁體中文或 English。')
    return language


def recognition_language(language: str) -> str:
    return 'en' if validate_language(language) == 'en' else 'zh'


class LanguageConversionError(ValueError):
    """A fixed, displayable error; never includes the user's text or a traceback."""


@lru_cache(maxsize=2)
def _converter(mode: str):
    try:
        from opencc import OpenCC
        return OpenCC(mode)
    except Exception:
        raise LanguageConversionError(
            '简繁转换组件不可用，请安装或修复 opencc-python-reimplemented。') from None


def _convert(text: str, mode: str) -> str:
    if not isinstance(text, str):
        raise TypeError('Chinese script conversion requires text')
    if not text:
        return text
    converter = _converter(mode)
    try:
        converted = converter.convert(text)
        if not isinstance(converted, str) or not converted:
            raise ValueError('invalid conversion result')
        return converted
    except Exception:
        raise LanguageConversionError(
            '简繁转换失败，请检查本地 OpenCC 安装后重试。') from None


def to_simplified(text: str) -> str:
    return _convert(text, 't2s')


def to_traditional(text: str) -> str:
    # s2t changes script, without s2twp's regional terminology substitutions.
    return _convert(text, 's2t')
