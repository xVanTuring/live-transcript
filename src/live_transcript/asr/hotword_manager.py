"""Dynamic hotword manager — extracts keywords from corrected text for streaming ASR boosting.

After 2nd-pass correction produces high-confidence text, this module:
1. Segments the text using jieba
2. Filters out stop words (particles, conjunctions, etc.)
3. Maintains a sliding window of recent keywords as hotwords
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common Chinese stop words: particles, conjunctions, pronouns, auxiliary verbs, etc.
_STOP_WORDS: set[str] = {
    # Particles
    "的", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "哈",
    "么", "嘛", "哪", "啥", "喂", "嘿", "唉", "哎",
    # Pronouns
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里",
    "什么", "怎么", "怎样", "如何", "哪个", "哪些", "谁",
    # Conjunctions & prepositions
    "和", "与", "或", "但", "但是", "然而", "而", "而且", "并且", "以及",
    "因为", "所以", "如果", "虽然", "不过", "于是", "因此",
    "在", "从", "到", "对", "向", "把", "被", "比", "跟", "给", "用",
    "按", "关于", "通过", "根据", "按照",
    # Auxiliary & common verbs
    "是", "有", "没有", "没", "不", "不是", "就是", "可以", "能", "会",
    "要", "想", "得", "地", "能够", "应该", "必须", "需要",
    "做", "去", "来", "说", "看", "知道", "觉得", "认为",
    # Adverbs
    "很", "非常", "特别", "比较", "最", "更", "都", "也", "就", "才",
    "又", "再", "还", "已经", "正在", "一直", "只", "只是",
    "大概", "可能", "当然", "其实", "然后", "接下来", "首先",
    # Measure words & numbers
    "个", "些", "种", "次", "下", "上", "中", "里", "一", "二", "三",
    "两", "几", "多", "少",
    # Fillers
    "就是说", "那么", "所以说", "对吧", "好的", "OK",
}


@dataclass
class HotwordManagerConfig:
    enabled: bool = False
    max_words: int = 50
    min_word_length: int = 2
    score: float = 1.5


class HotwordManager:
    """Maintains a sliding window of dynamically extracted hotwords."""

    def __init__(self, config: HotwordManagerConfig | None = None):
        self._config = config or HotwordManagerConfig()
        # OrderedDict as LRU: keys are words, values are insertion order
        self._words: OrderedDict[str, None] = OrderedDict()
        self._jieba_loaded = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def score(self) -> float:
        return self._config.score

    def update(self, text: str) -> None:
        """Extract keywords from corrected text and add to hotword window."""
        if not self._config.enabled or not text:
            return

        keywords = self._extract_keywords(text)
        for word in keywords:
            # Move to end (most recent) if already exists
            if word in self._words:
                self._words.move_to_end(word)
            else:
                self._words[word] = None

        # Trim to max window size
        while len(self._words) > self._config.max_words:
            self._words.popitem(last=False)

        if keywords:
            logger.debug(
                "Hotwords updated: +%d words, total=%d, new=%s",
                len(keywords), len(self._words),
                list(keywords)[:5],
            )

    def get_hotwords_str(self) -> str:
        """Return current hotwords as newline-separated string for sherpa-onnx."""
        if not self._words:
            return ""
        return "\n".join(self._words.keys())

    def clear(self) -> None:
        self._words.clear()

    def _extract_keywords(self, text: str) -> list[str]:
        """Segment text and filter to content words only."""
        self._ensure_jieba()
        import jieba.posseg as pseg

        # Clean text
        text = re.sub(r"<\|[^|]*\|>", "", text)  # Remove model tags
        text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)  # Keep Chinese + alphanumeric

        keywords = []
        for word, flag in pseg.cut(text):
            word = word.strip()
            if not word:
                continue
            if len(word) < self._config.min_word_length:
                continue
            if word.lower() in _STOP_WORDS:
                continue
            # Keep nouns, verbs, adjectives, proper nouns, English words
            # flag prefixes: n=noun, v=verb, a=adj, nr=person, ns=place, nt=org, eng=english
            if flag.startswith(("n", "v", "a", "eng", "x")):
                keywords.append(word)

        return keywords

    def _ensure_jieba(self) -> None:
        if not self._jieba_loaded:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            self._jieba_loaded = True
