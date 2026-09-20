# Evidence - Probe NIPC end-to-end Q4 2025 -> Q1 2026

Ciclo: IAE NIPC / Gate-NIPC.3 preparatorio.
Dictamen asociado: pendiente (Q-PROBE-1 a Q-PROBE-5).
Informe: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md

## Contenido

  probe_nipc_e2e.py                  Probe end-to-end Q4 -> Q1.
  probe_filer_continuity.py          Top 20 filer continuity.
  probe_filer_continuity_top50.py    Top 50 filer continuity.
  probe_reconcile_vanguard.py        Reconciliacion cuantitativa Vanguard.
  top50_output.txt                   Salida cruda del probe top 50.
  README.md                          Este fichero.
  HASHES.txt                         SHA-256 de cada artefacto.

## Reproducibilidad

    py docs/auditoria/iae/evidence/nipc_gate0_probe/probe_nipc_e2e.py
    py docs/auditoria/iae/evidence/nipc_gate0_probe/probe_filer_continuity.py
    py docs/auditoria/iae/evidence/nipc_gate0_probe/probe_filer_continuity_top50.py
    py docs/auditoria/iae/evidence/nipc_gate0_probe/probe_reconcile_vanguard.py

Capturar salida del probe top 50 (regenera top50_output.txt):

    py docs/auditoria/iae/evidence/nipc_gate0_probe/probe_filer_continuity_top50.py > top50_output.txt

Requiere acceso a:
  - D:/13f_probe/processed/2025Q4/ (7 parquets)
  - D:/13f_probe/processed/2026Q1/ (7 parquets)
  - D:/13f_probe/official_list_13f/13flist_2025Q4.txt
  - D:/13f_probe/official_list_13f/13flist_2026q1.txt
  - D:/Macro_Sectorial/ (repo con modulos NIPC)

Determinista. Sin estado aleatorio. Sin datetime.now().

## Salida de los probes

Ver seccion 3 del informe: elegibilidad SEC, resolucion de identidad,
NIPC por scope, coverage pairwise, status derivado.

top50_output.txt: salida cruda del probe top 50. Marcadores
verificables: CONTINUOUS_FILER: 50, FILER_DISCONTINUITY: 4,
Total: 54, pct_discontinuity: 0.0741. Los 4 CIKs Vanguard
(0000102909, 0000933478, 0002100119, 0002100121) aparecen en
detalle FILER_DISCONTINUITY. UNRESOLVED no se imprime cuando el
valor es 0 (Counter no incluye categorias vacias); coherente con
informe TOP50.

## Hallazgo principal

Reorganizacion de filer CIK del grupo Vanguard entre Q4 2025 y Q1 2026:
13F-HR unico -> 13F-NT padre + 2 filiales HR. El match key contractado
(C2) los trata como EXIT+NEW. No es bug. Documentado en seccion 4 del
informe.