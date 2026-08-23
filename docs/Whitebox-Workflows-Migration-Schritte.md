# Whitebox Workflows Migration Schritte

Stand: 2026-08-23

## Ziel
Schrittweise und risikoarm von direkter WhiteboxTools-Nutzung auf ein Runtime-Pattern mit whitebox_workflows umstellen, ohne bestehende Algorithmen auf einmal umzubauen.

## Leitprinzipien
- Kleine, isolierte Änderungen pro Schritt.
- Bestehende Funktionalität bleibt lauffähig (Fallback erhalten).
- Nach jedem Schritt: kurzer Syntax- und Funktionscheck.
- Erst nach stabilen Einzelschritten den nächsten Algorithmus migrieren.

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
- Bestehende WhiteboxTools-Ausführung wurde nicht verändert.
- Keine Umstellung in den Algorithmusklassen in diesem Schritt.

Nutzen:
- Fundament für kontrollierte Migration vorhanden.
- Wir können jetzt einzelne Algorithmen nacheinander umstellen.

## Nächster Schritt

### Schritt 2: Einen einzelnen Algorithmus optional über Adapter laufen lassen
Status: in Arbeit

Vorgehen:
1. Eine kleine, gut überschaubare Algorithmusklasse auswählen.
2. Dort einen optionalen Ausführungspfad via get_wbw_runtime(...) ergänzen.
3. Bestehenden WhiteboxTools-Pfad als Fallback behalten.
4. Verhalten (Outputs, Progress, Fehlertexte) gegen bisherigen Pfad vergleichen.

Abnahmekriterien:
- Algorithmus läuft weiterhin stabil.
- Keine Regression für bestehende Nutzer ohne whitebox_workflows.
- Logs/Feedback bleiben verständlich.

Aktueller Zwischenstand:
- Allgemeiner Wrapper ergänzt:
  - _execute_wbt_workflow(...)
  - Datei: topo_drain/core/topo_drain_core.py
  - Funktion: nutzt whitebox_workflows nur für freigegebene Tools und fällt sonst auf _execute_wbt zurück.
- Initiale globale Executor-Wahl ergänzt:
  - Neuer Init-Parameter: use_workflow_if_available=True
  - Neue Methode: _configure_wbt_executor()
  - Ergebnis: self.wbt_executor zeigt auf _execute_wbt_workflow oder _execute_wbt.
- Erster konkreter Einsatz:
  - breach_depressions_least_cost in extract_valleys verwendet jetzt den gewählten Executor via self.wbt_executor(...).

## Iterations-Checkliste (pro Schritt)
- Scope klar: nur 1 kleiner Teil geändert.
- Syntax/Editor-Fehler geprüft.
- Kein unbeabsichtigter Eingriff in andere Algorithmen.
- Kurze Doku hier aktualisiert.

## Offene Entscheidungen
- Reihenfolge der zu migrierenden Algorithmen.
- Wann Umschalten von optionalem Pfad auf Standardpfad sinnvoll ist.
- Ob langfristig WhiteboxTools- und whitebox_workflows-Pfad parallel bleiben sollen.

## Änderungsprotokoll
- 2026-08-23: Schritt 1 abgeschlossen (Adapter + Core-Zugriff).
- 2026-08-23: Schritt 2 gestartet (einzelner Call in extract_valleys auf runtime-first + Fallback umgestellt).
- 2026-08-23: Executor-Auswahl beim Initialisieren ergänzt (_configure_wbt_executor, use_workflow_if_available), _execute_wbt_auto wieder entfernt.
