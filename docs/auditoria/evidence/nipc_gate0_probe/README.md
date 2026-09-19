# Evidence - Probe NIPC end-to-end Q4 2025 -> Q1 2026

Ciclo: IAE NIPC / Gate-NIPC.3 preparatorio.
Dictamen asociado: pendiente (Q-PROBE-1 a Q-PROBE-5).
Informe: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md

## Contenido

  probe_nipc_e2e.py   Script reproducible del probe.
  HASHES.txt          SHA-256 del script.

## Reproducibilidad

    py docs/auditoria/evidence/nipc_gate0_probe/probe_nipc_e2e.py

Requiere acceso a:
  - D:/13f_probe/processed/2025Q4/ (7 parquets)
  - D:/13f_probe/processed/2026Q1/ (7 parquets)
  - D:/13f_probe/official_list_13f/13flist_2025Q4.txt
  - D:/13f_probe/official_list_13f/13flist_2026q1.txt
  - D:/Macro_Sectorial/ (repo con modulos NIPC)

Determinista. Sin estado aleatorio. Sin datetime.now().

## Salida del probe

Ver seccion 3 del informe: elegibilidad SEC, resolucion de identidad,
NIPC por scope, coverage pairwise, status derivado.

## Hallazgo principal

Reorganizacion de filer CIK del grupo Vanguard entre Q4 2025 y Q1 2026:
13F-HR unico -> 13F-NT padre + 2 filiales HR. El match key contractado
(C2) los trata como EXIT+NEW. No es bug. Documentado en seccion 4 del
informe.