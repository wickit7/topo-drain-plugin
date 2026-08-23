# -*- coding: utf-8 -*-
"""Small adapter around whitebox_workflows RuntimeSession.

This module is intentionally standalone so it can be introduced without
changing existing WhiteboxTools-based algorithm code paths.
"""

from __future__ import annotations

import json
from typing import Any


class WhiteboxWorkflowsRuntimeError(RuntimeError):
    """Raised when whitebox_workflows runtime initialization or execution fails."""


class WhiteboxWorkflowsRuntimeAdapter:
    """Thin runtime adapter for whitebox_workflows RuntimeSession.

    The adapter mirrors the execution style used by the Whitebox Workflows for
    QGIS plugin: JSON arguments in, JSON response out, with streamed progress
    events mapped to QGIS feedback when available.
    """

    def __init__(self, include_pro: bool = False, tier: str = "open") -> None:
        self.include_pro = bool(include_pro)
        self.tier = str(tier or "open").strip().lower() or "open"
        self._wbw = None
        self._session = None

    def _load_backend(self):
        if self._wbw is not None:
            return self._wbw

        try:
            import whitebox_workflows as wbw  # type: ignore
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                "whitebox_workflows is not installed in the current Python environment."
            ) from exc

        runtime_cls = getattr(wbw, "RuntimeSession", None)
        if runtime_cls is None:
            raise WhiteboxWorkflowsRuntimeError(
                "whitebox_workflows RuntimeSession is unavailable. A Next Gen runtime is required."
            )

        self._wbw = wbw
        return self._wbw

    def is_available(self) -> bool:
        try:
            self._load_backend()
            return True
        except Exception:
            return False

    def get_session(self):
        if self._session is not None:
            return self._session

        wbw = self._load_backend()
        try:
            self._session = wbw.RuntimeSession(
                include_pro=self.include_pro,
                tier=self.tier,
            )
            return self._session
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                f"Failed to create whitebox_workflows RuntimeSession: {exc}"
            ) from exc

    def get_capabilities(self) -> dict[str, Any]:
        session = self.get_session()
        try:
            payload = session.get_runtime_capabilities_json()
            return json.loads(payload) if isinstance(payload, str) else dict(payload)
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                f"Could not read runtime capabilities: {exc}"
            ) from exc

    def run_tool_json_stream(
        self,
        tool_id: str,
        args: dict[str, Any],
        feedback=None,
    ) -> dict[str, Any]:
        session = self.get_session()

        def _stream_callback(evt: Any) -> None:
            if feedback is None:
                return

            event_obj: Any = evt
            if isinstance(evt, str):
                try:
                    event_obj = json.loads(evt)
                except Exception:
                    event_obj = {"type": "message", "message": evt}

            if not isinstance(event_obj, dict):
                return

            event_type = str(event_obj.get("type", "message")).lower()
            message = str(event_obj.get("message") or event_obj.get("text") or "").strip()

            if event_type == "progress":
                try:
                    pct_raw = event_obj.get("percent")
                    if pct_raw is not None:
                        feedback.setProgress(float(pct_raw))
                except Exception:
                    pass
                if message:
                    try:
                        feedback.pushInfo(message)
                    except Exception:
                        pass
                return

            if event_type in {"warning", "warn"}:
                if message:
                    try:
                        feedback.pushWarning(message)
                    except Exception:
                        pass
                return

            if event_type in {"error", "fatal", "critical"}:
                if message:
                    try:
                        feedback.reportError(message)
                    except Exception:
                        pass
                return

            if message:
                try:
                    feedback.pushInfo(message)
                except Exception:
                    pass

        try:
            response_raw = session.run_tool_json_stream(
                str(tool_id),
                json.dumps(args),
                _stream_callback,
            )
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                f"whitebox_workflows execution failed for tool '{tool_id}': {exc}"
            ) from exc

        if response_raw is None:
            return {}

        try:
            response = json.loads(response_raw) if isinstance(response_raw, str) else response_raw
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                f"Tool '{tool_id}' returned invalid JSON payload: {exc}"
            ) from exc

        if isinstance(response, dict):
            return response
        return {"result": response}
