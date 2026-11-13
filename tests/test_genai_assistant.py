"""Offline contract tests for the Gemini resume-assistant integration."""

from types import SimpleNamespace

from modules.genai_assistant import GenAIAssistant


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def configured_assistant(response_text):
    assistant = GenAIAssistant(api_key="test-key", model_name="gemini-test-flash")
    models = FakeModels(response_text)
    assistant.client = SimpleNamespace(models=models)
    return assistant, models


def test_missing_key_is_safe():
    assistant = GenAIAssistant(api_key="")
    assert assistant.test_connection()["success"] is False
    assert "GEMINI_API_KEY" in assistant.configuration_message()


def test_question_uses_only_uploaded_resume_context():
    assistant, models = configured_assistant("The uploaded resume lists Python.")
    result = assistant.answer_question("Skills: Python", "What skills are listed?")
    assert result["success"] is True
    assert result["answer"] == "The uploaded resume lists Python."
    assert "UPLOADED RESUME" in models.calls[0]["contents"]
    assert "Skills: Python" in models.calls[0]["contents"]


def test_structured_insights_parse_json():
    assistant, _ = configured_assistant('{"summary":"Python engineer","skills":["Python"],"education":[],"experience":[],"projects":[],"improvements":[]}')
    result = assistant.generate_structured_insights("Skills: Python")
    assert result["success"] is True
    assert result["data"]["skills"] == ["Python"]


def test_invalid_key_error_is_not_exposed():
    message = GenAIAssistant._error_message(RuntimeError("401 invalid API key"))
    assert "authenticate" in message
    assert "invalid API key" not in message.lower()
