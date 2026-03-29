"""Shared LLM factory for MARA agent nodes.

All nodes that call HuggingFace Inference Providers import ``make_llm`` from
here so that provider configuration lives in one place.

``ChatHuggingFace`` is re-exported so callers that need it for type
annotations don't import ``langchain_huggingface`` directly.
"""

from __future__ import annotations

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

__all__ = ["ChatHuggingFace", "make_llm"]


def make_llm(
    model: str,
    hf_token: str,
    max_new_tokens: int,
    provider: str = "featherless-ai",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> ChatHuggingFace:
    """Instantiate a ``ChatHuggingFace`` client via HuggingFace Inference Providers.

    Args:
        model:          HuggingFace model repo ID.
        hf_token:       HuggingFace Hub API token.
        max_new_tokens: Token budget for the completion.
        provider:       HF inference provider name. Defaults to ``"featherless-ai"``.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling probability.
        top_k:          Top-k sampling.

    Returns:
        A ``ChatHuggingFace`` instance ready for ``.invoke()`` or ``.ainvoke()``.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=max_new_tokens,
        provider=provider,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return ChatHuggingFace(llm=endpoint)
