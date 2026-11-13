"""Grounded Google Gemini support for uploaded SmartResume documents."""

import json
import os
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GOOGLE_GENAI = False

from config.settings import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """You are SmartResume AI, an evidence-grounded resume assistant.
Use only facts explicitly present in the uploaded resume text. Never invent a
qualification, employer, date, skill, degree, project, certification, or contact
detail. When information is unavailable, say exactly: "Not mentioned in the
uploaded resume". Keep each response limited to the supplied resume."""

PREFERRED_FLASH_MODELS = (
    "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite",
    "gemini-2.5-flash", "gemini-2.0-flash",
)


class GenAIAssistant:
    """Failure-safe wrapper around the current google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key if api_key is not None else (os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY)
        self.model_name = model_name if model_name is not None else (os.getenv("GEMINI_MODEL", "") or GEMINI_MODEL)
        self.client = None
        self._resolved_model: Optional[str] = None
        if self.is_configured():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_configured(self) -> bool:
        return bool(HAS_GOOGLE_GENAI and self.api_key and not self.api_key.startswith("your_"))

    def configuration_message(self) -> str:
        if not HAS_GOOGLE_GENAI:
            return "Gemini support is unavailable. Install google-genai in the project environment."
        if not self.api_key or self.api_key.startswith("your_"):
            return "Gemini is not configured. Add GEMINI_API_KEY to your local .env file and restart the app."
        return "Gemini is unavailable. Check the local API configuration and try again."

    @staticmethod
    def _error_message(error: Exception) -> str:
        message = str(error).lower()
        if any(token in message for token in ("api key", "unauthenticated", "permission_denied", "401", "403")):
            return "Gemini could not authenticate the API key. Check GEMINI_API_KEY in .env."
        if any(token in message for token in ("quota", "rate limit", "resource_exhausted", "429")):
            return "Gemini quota or rate limit reached. Please wait and try again."
        if any(token in message for token in ("timeout", "network", "connection", "unavailable", "dns")):
            return "Gemini is temporarily unreachable. Check your network connection and try again."
        if any(token in message for token in ("model", "not_found", "404")):
            return "The configured Gemini model is unavailable. Leave GEMINI_MODEL blank or select a supported Flash model."
        if "empty response" in message:
            return "Gemini returned an empty response. Please try again."
        return "Gemini returned an unexpected error. Please try again."

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        if self.model_name:
            self._resolved_model = self.model_name
            return self._resolved_model
        available = [
            model.name.removeprefix("models/")
            for model in self.client.models.list()
            if "generateContent" in (getattr(model, "supported_actions", None) or [])
        ]
        for candidate in PREFERRED_FLASH_MODELS:
            if candidate in available:
                self._resolved_model = candidate
                return candidate
        flash_models = [name for name in available if "flash" in name.lower()]
        if flash_models:
            self._resolved_model = flash_models[0]
            return self._resolved_model
        raise RuntimeError("No Gemini Flash model is available for this account.")

    def test_connection(self) -> Dict[str, Any]:
        if not self.client:
            return {"success": False, "error": self.configuration_message()}
        try:
            model = self._resolve_model()
            response = self.client.models.generate_content(
                model=model,
                contents="Reply with exactly: OK",
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=5),
            )
            if not (response.text or "").strip():
                raise RuntimeError("Gemini returned an empty response")
            return {"success": True, "model": model}
        except Exception as error:
            return {"success": False, "error": self._error_message(error)}

    def _generate(self, resume_text: str, prompt: str, *, json_output: bool = False) -> Dict[str, Any]:
        if not self.client:
            return {"success": False, "error": self.configuration_message(), "text": ""}
        try:
            config_kwargs: Dict[str, Any] = {"system_instruction": SYSTEM_PROMPT, "temperature": 0.2, "max_output_tokens": 1200}
            if json_output:
                config_kwargs["response_mime_type"] = "application/json"
            response = self.client.models.generate_content(
                model=self._resolve_model(),
                contents=f"UPLOADED RESUME:\n---\n{resume_text}\n---\n\n{prompt}",
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return {"success": True, "text": text, "model": self._resolved_model}
        except Exception as error:
            return {"success": False, "error": self._error_message(error), "text": ""}

    def answer_question(self, resume_text: str, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        result = self._generate(resume_text, f"Answer this question about the uploaded resume: {question}")
        return {**result, "answer": result.get("text", "")}

    def generate_summary(self, resume_text: str) -> Dict[str, Any]:
        result = self._generate(resume_text, "Write a concise candidate summary with headings: Professional overview, Skills and education, Experience and projects. Cite only uploaded-resume facts.")
        return {**result, "summary": result.get("text", "")}

    def suggest_improvements(self, resume_text: str) -> Dict[str, Any]:
        result = self._generate(resume_text, "Provide resume improvements under: Missing information, Impact and clarity, Formatting, and ATS keywords. Identify absent details as Not mentioned in the uploaded resume.")
        return {**result, "critique": result.get("text", "")}

    def generate_structured_insights(self, resume_text: str) -> Dict[str, Any]:
        prompt = """Return JSON only with exactly: {"summary":"", "skills":[], "education":[], "experience":[], "projects":[], "improvements":[]}. Use strings in arrays. Use "Not mentioned in the uploaded resume" for unavailable scalar information and [] for unavailable list information."""
        result = self._generate(resume_text, prompt, json_output=True)
        if not result["success"]:
            return {**result, "data": None}
        try:
            return {**result, "data": json.loads(result["text"])}
        except json.JSONDecodeError:
            return {"success": False, "error": "Gemini returned invalid structured data. Please try again.", "data": None}
