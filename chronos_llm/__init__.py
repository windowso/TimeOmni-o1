"""Chronos-2 + LLM: a unified model for time-series understanding and forecasting.

Built on top of the chronos-forecasting repository, it connects the Chronos-2 time-series backbone and the
Qwen3.5 LLM in both directions -- a sliding-window Q-former (history -> LLM) and gated cross-attention
(LLM -> Chronos-2) -- for time-series understanding and forecasting with reasoning in text.
"""
