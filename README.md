\# Radar de Rotación Sectorial v4.3



Sistema determinista y descriptivo para el análisis de rotación sectorial basado en flujos institucionales, regímenes macro y liderazgo relativo.



\*\*No es un sistema de trading automático.\*\*  

No genera señales de compra/venta ni optimiza parámetros con fines predictivos.



\## Arquitectura general



OHLCV → Regímenes → Motores → Indicadores → Scores → Reporte Markdown  

&#x20;                            ↓  

&#x20;                      Validation Gate 10/10



\## Flujo diario (`run.py`)



1\. Descarga de datos de mercado (Yahoo Finance + proveedores de respaldo).  

2\. Validación de calidad y cobertura.  

3\. Cálculo de regímenes: financiero, liquidez, volatilidad, macro, sectorial.  

4\. Motores táctico y estructural.  

5\. Indicadores: momentum, breadth, persistence, SLPM, MTE, opciones, dark pools, flujos primarios.  

6\. Selección de líderes sectoriales e internacionales.  

7\. Generación de reporte Markdown.  

8\. Validation Gate → si falla, el sistema se detiene (`sys.exit(1)`).  



\## Capas de flujo implementadas



\- \*\*ETF\_PRIMARY\_FLOW\*\* (diario): SSGA, BlackRock DAXEX/ISF.L/IWM, Amundi LYXI.  

\- \*\*CFTC\_POSITION\_FLOW\*\* (semanal): futuros financieros, frescura 30 días.  

\- \*\*SEC\_POSITION\_FLOW\*\* (trimestral): N-PORT.  

\- \*\*QQQ NPORT-P FLOW\*\* (trimestral): Item B.6 Sales/Redemptions.  

\- \*\*FLOW\_PROXY\*\* (diario): precio/volumen, no es flujo primario.  

\- \*\*FLOW\_SYNTHESIS\*\* (descriptivo): concordancia de signos.  



\*\*No se mezclan capas de flujo ni se construyen superindicadores predictivos.\*\*



\## Validación



La Validation Gate comprueba 10 condiciones mínimas de integridad y frescura. El sistema solo se considera operativo con 10/10.



\## Uso local



```powershell

py run.py

py validation\\run\_all\_audits.py

