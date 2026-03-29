"""Tests for mara/llm.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mara.llm import ChatHuggingFace, make_llm


def test_make_llm_returns_chat_hugging_face():
    """make_llm constructs ChatHuggingFace wrapping a HuggingFaceEndpoint."""
    fake_endpoint = MagicMock()
    fake_chat = MagicMock(spec=ChatHuggingFace)

    with (
        patch("mara.llm.HuggingFaceEndpoint", return_value=fake_endpoint) as mock_ep,
        patch("mara.llm.ChatHuggingFace", return_value=fake_chat) as mock_chat,
    ):
        result = make_llm(
            model="org/model",
            hf_token="tok",
            max_new_tokens=512,
        )

    mock_ep.assert_called_once_with(
        repo_id="org/model",
        task="text-generation",
        huggingfacehub_api_token="tok",
        max_new_tokens=512,
        provider="featherless-ai",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    mock_chat.assert_called_once_with(llm=fake_endpoint)
    assert result is fake_chat


def test_make_llm_custom_provider_and_sampling():
    """make_llm forwards non-default provider and sampling params."""
    fake_endpoint = MagicMock()

    with (
        patch("mara.llm.HuggingFaceEndpoint", return_value=fake_endpoint) as mock_ep,
        patch("mara.llm.ChatHuggingFace"),
    ):
        make_llm(
            model="m",
            hf_token="t",
            max_new_tokens=256,
            provider="hf-inference",
            temperature=0.1,
            top_p=0.9,
            top_k=50,
        )

    _, kwargs = mock_ep.call_args
    assert kwargs["provider"] == "hf-inference"
    assert kwargs["temperature"] == 0.1
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50
