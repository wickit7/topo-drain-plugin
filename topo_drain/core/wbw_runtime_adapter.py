# -*- coding: utf-8 -*-
"""Small adapter around whitebox_workflows RuntimeSession.

This module is intentionally standalone so it can be introduced without
changing existing WhiteboxTools-based algorithm code paths.
"""

from __future__ import annotations

import logging
import gc
import json
import threading
from typing import Any

_log = logging.getLogger(__name__)


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
        self._session_thread_id = None
        self._teardown_logged = False

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

    def _create_session(self):
        wbw = self._load_backend()
        try:
            return wbw.RuntimeSession(
                include_pro=self.include_pro,
                tier=self.tier,
            )
        except Exception as exc:
            raise WhiteboxWorkflowsRuntimeError(
                f"Failed to create whitebox_workflows RuntimeSession: {exc}"
            ) from exc

    @staticmethod
    def _dispose_session(session: Any) -> None:
        if session is None:
            return

        # Different runtime builds may expose different teardown methods.
        for method_name in ("close", "shutdown", "dispose", "stop"):
            method = getattr(session, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as e:
                    _log.debug("[WbwRuntimeAdapter] Exception in teardown method %s: %s", method_name, e)
                break

    @staticmethod
    def _detect_teardown_methods(session: Any) -> tuple[str | None, list[str]]:
        if session is None:
            return None, []

        available: list[str] = []
        for method_name in ("close", "shutdown", "dispose", "stop"):
            method = getattr(session, method_name, None)
            if callable(method):
                available.append(method_name)

        preferred = available[0] if available else None
        return preferred, available

    def get_session(self, refresh: bool = False):
        if refresh and self._session is not None:
            old_session = self._session
            self._dispose_session(old_session)
            self._session = None
            # Some RuntimeSession builds expose no explicit close/dispose API.
            # gc.collect() (full) stops every thread in the process, including the Qt
            # main/GUI thread, and was freezing QGIS on every algorithm run; gen-0-only
            # is enough to break cyclic refs to the just-dereferenced session.
            del old_session
            gc.collect(0)

        if refresh or self._session is None:
            self._session = self._create_session()
            self._session_thread_id = threading.get_ident()
        return self._session

    def reset_session(self) -> None:
        """Drop the cached session so the next call creates a fresh RuntimeSession."""
        old_session = self._session
        if old_session is None:
            return  # Nothing to dispose; avoid an unnecessary GC pause on every run.
        self._dispose_session(old_session)
        self._session = None
        self._session_thread_id = None
        # Some RuntimeSession builds expose no explicit close/dispose API.
        # gc.collect() (full) stops every thread in the process, including the Qt
        # main/GUI thread, and was freezing QGIS on every algorithm run; gen-0-only
        # is enough to break cyclic refs to the just-dereferenced session.
        del old_session
        gc.collect(0)

    def reset_session_if_stale(self) -> None:
        """Reset the cached session only if it belongs to a different thread.

        Constructing a RuntimeSession is expensive and can block QGIS's main GUI
        thread (native init doesn't release the GIL). Reusing a session across a
        terminated QThread crashes on Windows, so only reset when the owning
        thread actually changed instead of unconditionally on every run.
        """
        if self._session is not None and self._session_thread_id == threading.get_ident():
            return
        self.reset_session()

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
        report_progress: bool = True,
    ) -> dict[str, Any]:
        if feedback is not None:
            try:
                if feedback.isCanceled():
                    raise WhiteboxWorkflowsRuntimeError("Process cancelled by user.")
            except AttributeError:
                pass

        cancel_requested = {"value": False}

        def _stream_callback(evt: Any) -> None:
            if feedback is None:
                return

            try:
                if feedback.isCanceled():
                    # Avoid raising from callback frames that may cross FFI boundaries.
                    cancel_requested["value"] = True
                    return
            except AttributeError:
                pass

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
                if report_progress:
                    try:
                        pct_raw = event_obj.get("percent")
                        if pct_raw is not None:
                            feedback.setProgress(float(pct_raw))
                    except Exception as exc:
                        _log.debug("feedback.setProgress failed: %s", exc)
                    if message:
                        try:
                            feedback.pushInfo(message)
                        except Exception as exc:
                            _log.debug("feedback.pushInfo failed: %s", exc)
                return

            if event_type in {"warning", "warn"}:
                if message:
                    try:
                        feedback.pushWarning(message)
                    except Exception as exc:
                        _log.debug("feedback.pushWarning failed: %s", exc)
                return

            if event_type in {"error", "fatal", "critical"}:
                if message:
                    try:
                        feedback.reportError(message)
                    except Exception as exc:
                        _log.debug("feedback.reportError failed: %s", exc)
                return

            if message and report_progress:
                try:
                    feedback.pushInfo(message)
                except Exception as exc:
                    _log.debug("feedback.pushInfo failed: %s", exc)

        def _run_once() -> Any:
            session = self.get_session(refresh=False)
            if feedback is not None and not self._teardown_logged:
                preferred, available = self._detect_teardown_methods(session)
                try:
                    feedback.pushInfo(
                        "[WBTWorkflow] RuntimeSession teardown methods "
                        f"available={available if available else ['none']}, "
                        f"preferred={preferred if preferred else 'none'}"
                    )
                except Exception as exc:
                    _log.debug("feedback.pushInfo failed: %s", exc)
                self._teardown_logged = True
            return session.run_tool_json_stream(
                str(tool_id),
                json.dumps(args),
                _stream_callback,
            )

        try:
            response_raw = _run_once()
        except Exception as first_exc:
            if cancel_requested["value"]:
                raise WhiteboxWorkflowsRuntimeError("Process cancelled by user.") from first_exc
            # Reset and retry once with a fresh session. This mirrors robust
            # session recovery behavior and avoids stale session crashes.
            self.reset_session()
            try:
                response_raw = _run_once()
            except Exception as exc:
                if feedback is not None:
                    try:
                        if feedback.isCanceled() or cancel_requested["value"]:
                            raise WhiteboxWorkflowsRuntimeError("Process cancelled by user.") from exc
                    except AttributeError:
                        pass
                raise WhiteboxWorkflowsRuntimeError(
                    f"whitebox_workflows execution failed for tool '{tool_id}': {exc}"
                ) from exc

        if cancel_requested["value"]:
            raise WhiteboxWorkflowsRuntimeError("Process cancelled by user.")

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
