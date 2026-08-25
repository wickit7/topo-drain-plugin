# Whitebox Workflows Migration Schritte

Stand: 2026-08-25

## Ziel
Schrittweise und risikoarm auf whitebox_workflows als Laufzeit-Backend umstellen und verbleibende Unterschiede bei Tool-Verhalten/Performance stabilisieren.

## Leitprinzipien
- Kleine, isolierte Änderungen pro Schritt.
- Nach jedem Schritt: kurzer Syntax- und Funktionscheck.
- Fehlerbilder zuerst reproduzierbar machen, dann dauerhaft fixen.
- Bei instabilen Runtime-Calls: zunächst manueller Debug-Pfad in QGIS ermöglichen.

## Bisher umgesetzt

### Schritt 1: Runtime-Adapter als isolierter Baustein
Status: erledigt

Änderungen:
- Neue Adapter-Datei ergänzt:
  - topo_drain/core/wbw_runtime_adapter.py
- Core um optionalen Zugriff auf Adapter erweitert:
  - topo_drain/core/topo_drain_core.py
  - Neuer, gecachter Zugriff: get_wbw_runtime(include_pro=False, tier="open")

Wichtig:
- Bestehende WhiteboxTools-Ausführung wurde in diesem Schritt noch nicht verändert.
- Keine Umstellung in den Algorithmusklassen in diesem Schritt.

Nutzen:
- Fundament für kontrollierte Migration vorhanden.
- Wir können jetzt einzelne Algorithmen nacheinander umstellen.

### Schritt 2: Executor-Umstellung im Core
Status: erledigt

Änderungen:
- Globale Executor-Auswahl im Core ergänzt:
  - `self.wbt_executor`
  - `_configure_wbt_executor()`
- Ausführung auf whitebox_workflows zentralisiert.
- Runtime-Verifikation beim Start ergänzt.

Ergebnis:
- Core nutzt jetzt den Workflow-Pfad konsistent.

### Schritt 3: Parameter-/Tool-ID-Abgleich auf Workflow-API
Status: größtenteils erledigt

Korrigierte Mismatches (Auszug):
- `d8_flow_accumulation` -> `d8_flow_accum`
- `VectorStreamNetworkAnalysis` -> `vector_stream_network_analysis`
- diverse Parameternamen auf Workflow-Signaturen angepasst.

### Schritt 4: Laufzeit-Debugging ExtractValleys (letzter Schritt)
Status: in Arbeit

Bestätigter Stand aus QGIS:
- `fix_dangling_arcs` funktioniert.
- `vector_stream_network_analysis` im alten Plugin **WhiteboxTools for QGIS** läuft mit denselben Daten/Parametern erfolgreich durch.
- `vector_stream_network_analysis` im neuen Workflow-Pfad zeigt weiterhin abweichendes Laufzeitverhalten.

Bekannte Ursache/Beobachtung:
- Unterschied scheint backend-/wrapper-spezifisch (klassischer WBT-Call erfolgreich, Workflow-Runtime abweichend).
- Frühere Parsing-Fehler (`vector` als Pseudo-Pfad) wurden behoben.

Aktuelle Maßnahme im Code:
- Letzter Schritt in `extract_valleys` wieder aktiviert.
- Finaler Aufruf versucht zuerst workflow-native Parameter (`max_ridge_cutting_height`, `snap_distance`) und danach legacy-kompatible Parameter (`cutting_height`, `snap`).
- Bei Fehlschlag werden beide Versuche klar im Log gemeldet.

Zusätzliche Stabilitätsmaßnahme:
- Kein klassisches Backend-Fallback mehr (Policy: workflows-only für Enduser ohne WhiteboxTools-Installation).
- Stattdessen ist der finale Schritt (`vector_stream_network_analysis`) im Algorithmus als optionaler Schalter verfügbar und standardmäßig deaktiviert.
- In der Default-Konfiguration liefert `Create Valleys` stabil das verlinkte Stream-Netz (`streams_linked`) als Ergebnis.

## Aktueller manueller QGIS-Testpfad

1. `Create Valleys` ausführen bis zur Meldung, dass finaler Netzwerk-Schritt übersprungen wurde.
2. `streams_linked` als manuelle Eingabe verwenden.
3. Optional `fix_dangling_arcs` ausführen (bereits als funktionsfähig bestätigt).
4. `vector_stream_network_analysis` manuell in QGIS starten und Verhalten je Parameter dokumentieren.

Hinweis:
- Der Punkt 1 ("übersprungen") war ein temporärer Zustand und wurde wieder entfernt.

### Schritt 5: Stabilisierung CreateKeylines (wiederholte Ausführung)
Status: erledigt

Änderungen:
- `create_keylines` erzeugt temporäre Artefakte pro Lauf mit eindeutigen IDs (`run_uid`) und `_get_constant_slope_line` pro Funktionsaufruf (`call_uid`), um Kollisionen bei wiederholten Ausführungen zu vermeiden.
- Runtime-Adapter stabilisiert:
  - Fortschritts-Events werden nur noch gesetzt, wenn `report_progress=True`.
  - Session-Reuse mit gezieltem Reset bei Fehlern (`reset_session()` + einmaliger Retry).
  - Teardown-Methoden der RuntimeSession werden einmalig protokolliert.
- `Create Keylines` läuft in einer isolierten `TopoDrainCore`-Instanz pro Ausführung, inkl. Session-Drop/GC nach der Ausführung.

Ergebnis:
- Wiederholte Ausführungen von `Create Keylines` sind in QGIS stabilisiert (deutlich weniger bzw. keine direkten Abstürze in der gemeldeten Testserie).
- Haupt-Fortschrittsanzeige bleibt konsistent und wird nicht mehr durch innere Workflow-Events zurückgesetzt.

## Offene Punkte

- Reproduzierbaren Minimalfall dokumentieren, der den Unterschied zwischen klassischem WBT-Call und Workflow-Runtime zeigt.
- Bei weiterem Fehlverhalten gezielt Upstream-Issue für whitebox_workflows mit Datenausschnitt, Parametern und Logs erstellen.
- Optional: finalen Workflow-Schritt erst nach Upstream-Fix wieder standardmäßig aktivieren.
- Optional: Feingranulare Log-Verbosity für innere Slope-Iterationen als UI-Schalter ergänzen, um Processing-Logs bei großen Läufen kompakter zu halten.

## Iterations-Checkliste (pro Schritt)
- Scope klar: nur 1 kleiner Teil geändert.
- Syntax/Editor-Fehler geprüft.
- Kein unbeabsichtigter Eingriff in andere Algorithmen.
- Kurze Doku hier aktualisiert.

## Änderungsprotokoll
- 2026-08-23: Schritt 1 abgeschlossen (Adapter + Core-Zugriff).
- 2026-08-23: Executor-Auswahl beim Initialisieren ergänzt (_configure_wbt_executor).
- 2026-08-23: Tool-IDs/Parameter in mehreren Workflow-Calls angepasst.
- 2026-08-24: `fix_dangling_arcs`-Integration ergänzt; Parameter `snap` korrigiert.
- 2026-08-24: Parsing-Fix für Workflow-Outputs (nur echte Dateipfade verwenden).
- 2026-08-24: Validierung: `VectorStreamNetworkAnalysis` im alten Plugin (`WhiteboxTools for QGIS`) erfolgreich mit denselben Inputs/Parametern.
- 2026-08-24: Finale `vector_stream_network_analysis` in `extract_valleys` wieder aktiviert; zweistufiger Parameter-Versuch (workflow-native, dann legacy-kompatibel).
- 2026-08-24: Auf workflows-only zurückgestellt (kein klassisches Backend-Fallback mehr).
- 2026-08-24: Neuer Algorithmus-Schalter `Run final network analysis` ergänzt; standardmäßig deaktiviert zur Stabilisierung.
- 2026-08-25: Legacy-Schalter/Codepfade entfernt (`use_workflow_if_available`, `wbt_executor_name`, `_run_cost_distance_workflow_direct`).
- 2026-08-25: Fortschritts-Regression behoben (`report_progress=False` wird im Runtime-Adapter respektiert).
- 2026-08-25: Stabilisierung wiederholter `Create Keylines`-Läufe durch isolierte Core-Instanz pro Ausführung und expliziten Session-Drop/GC nach Run.
