'''Tests contractuales P66 (§14.3 reformulada).

Trazabilidad: ciclo P66, GO CONTRACTUAL #40 (commit 6e698ca).
Traslado contractual: commit 3a233b4.

Capa A (PASS): literales contractuales presentes en el .md.
Capa B (PASS): traslado no rompe P65 (no regresion).
Capa C (XFAIL strict=False): casos semanticos pendientes de
implementacion. Razon del xfail: '14.3.X implementation pending
(GO #40 step 2)'. Cuando el ciclo de implementacion exponga el
codigo contractual: quitar el decorador xfail y sustituir el
raise AssertionError por la llamada + assert real.

NO tocan codigo productivo.
'''
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRATO = REPO_ROOT / 'docs' / 'auditoria' / 'iae' / 'NIPC_CONTRATOS_SEMANTICOS_v1.md'


def _contrato_texto():
    return CONTRATO.read_text(encoding='utf-8-sig')


# ==================== Capa A: integridad documental ====================


def test_p66_contrato_existe():
    assert CONTRATO.exists(), 'NIPC_CONTRATOS_SEMANTICOS_v1.md no encontrado'


def test_p66_siete_subsecciones_14_3():
    texto = _contrato_texto()
    for i in range(1, 8):
        assert '#### 14.3.{0}.'.format(i) in texto


def test_p66_paso0_presente():
    texto = _contrato_texto()
    assert 'PASO 0' in texto
    assert 'completitud del scope' in texto


def test_p66_candidate_a_definida():
    texto = _contrato_texto()
    assert 'candidate_A' in texto
    assert 'resolved_ciks' in texto

def test_p66_r3_tri_state_literal():
    texto = _contrato_texto()
    # Literales reales del contrato 14.3.1 + 14.3 (booleano L3).
    for estado in ('R3 = FALSE', 'R3 = N/D', 'R3(B, period) == TRUE'):
        assert estado in texto


def test_p66_r4_estados_y_precedencia():
    texto = _contrato_texto()
    for estado in ('MATCH', 'NO_MATCH', 'N/D', 'CONFLICT'):
        assert estado in texto
    assert 'CONFLICT PREVALECE sobre MATCH' in texto
    assert 'N/D != NO_MATCH' in texto


def test_p66_familias_notice_combination():
    texto = _contrato_texto()
    assert '13F NOTICE' in texto
    assert '13F COMBINATION REPORT' in texto
    assert 'NOTICE family' in texto
    assert 'COMBINATION family' in texto


def test_p66_formnum_prefijo_y_padding():
    texto = _contrato_texto()
    assert '028' in texto
    assert 'padding a 5' in texto


def test_p66_l3_booleano_explicito():
    texto = _contrato_texto()
    assert 'L3' in texto
    for r in ('R1', 'R2', 'R3', 'R4', 'R5'):
        assert r in texto


def test_p66_clausula_cierre_14_3_7():
    texto = _contrato_texto()
    assert 'BLOQUEO MATERIAL' in texto
    assert 'RECOMENDACION DIFERIBLE' in texto

# ==================== Capa B: no regresion P65 ====================


def test_p66_p65_transiciones_intactas():
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.TRANSITION_NULL == 'NULL'
    assert rd.TRANSITION_HANDOFF == 'HANDOFF'
    assert rd.TRANSITION_DUPLICATE_REMOVED == 'DUPLICATE_REMOVED'
    assert rd.TRANSITION_DUPLICATION_UNRESOLVED == 'DUPLICATION_UNRESOLVED'
    assert len(rd.ALL_TRANSITIONS) == 4


def test_p66_p65_dedup_reasons_intactos():
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.DEDUP_REASON_INTRA_PERIOD_DUP == 'INTRA_PERIOD_DUP'
    assert rd.DEDUP_REASON_POSITION_HANDOFF == 'POSITION_SCOPED_HANDOFF'
    assert rd.DEDUP_REASON_REPORTING_CONFLICT == 'REPORTING_CONFLICT'
    assert rd.DEDUP_REASON_OVERLAP_UNRESOLVED == 'REPORTING_OVERLAP_UNRESOLVED'


def test_p66_p65_forbidden_terms_no_economic_owner():
    from src.institutional_accumulation.sec_13f.identity import relationships as rel
    assert hasattr(rel, 'FORBIDDEN_TERMS')
    joined = ' '.join(str(t) for t in rel.FORBIDDEN_TERMS)
    assert 'economic_owner' in joined

# ==================== Capa C: semantica pendiente (§14.3) ====================
# Los tests de esta capa documentan el comportamiento contractual P66
# PENDIENTE de implementacion. El decorador xfail documenta el estado;
# cuando el ciclo de implementacion (GO #40 step 2) exponga el codigo
# contractual, se elimina el decorador y se sustituye el raise por la
# llamada + assert real. El test debe pasar.

XFAIL = '14.3 implementation pending (GO #40 step 2)'


def test_p66_r3_true_un_base_notice():
    '''14.3.1: 1 base NOTICE + scope verificado -> R3 = TRUE.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    filings = [{
        "PERIODOFREPORT": "2026-03-31",
        "SUBMISSIONTYPE": "13F-NT",
        "REPORTTYPE": "13F NOTICE",
        "AMENDMENTNO": None, "AMENDMENTTYPE": None,
    }]
    assert rd.resolve_r3(filings, period="2026-03-31",
                         scope_completeness_verified=True) == "TRUE"


def test_p66_r3_false_scope_completo_sin_filings():
    '''14.3.1 PASO 0+4: scope completo verificado + 0 filings -> R3 = FALSE.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.resolve_r3([], period="2026-03-31",
                         scope_completeness_verified=True) == "FALSE"


def test_p66_r3_nd_scope_no_demostrable():
    '''14.3.1 PASO 0: scope no demostrable -> R3 = N/D.

    Regla: "no aparece" != "no existe".
    '''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.resolve_r3([], period="2026-03-31",
                         scope_completeness_verified=False) == "N/D"


def test_p66_r3_nd_familias_mixtas():
    '''14.3.1 PASO 3: NT + HR COMBINATION -> R3 = N/D (Gate 0.9).'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    filings = [
        {"PERIODOFREPORT": "2026-03-31", "SUBMISSIONTYPE": "13F-NT",
         "REPORTTYPE": "13F NOTICE", "AMENDMENTNO": None, "AMENDMENTTYPE": None},
        {"PERIODOFREPORT": "2026-03-31", "SUBMISSIONTYPE": "13F-HR",
         "REPORTTYPE": "13F COMBINATION REPORT", "AMENDMENTNO": None, "AMENDMENTTYPE": None},
    ]
    assert rd.resolve_r3(filings, period="2026-03-31",
                         scope_completeness_verified=True) == "N/D"


def test_p66_r3_nd_multiples_bases():
    '''14.3.1 PASO 4: >1 BASE -> R3 = N/D (no primero/ultimo/MAX).'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    filings = [
        {"PERIODOFREPORT": "2026-03-31", "SUBMISSIONTYPE": "13F-NT",
         "REPORTTYPE": "13F NOTICE", "AMENDMENTNO": None, "AMENDMENTTYPE": None},
        {"PERIODOFREPORT": "2026-03-31", "SUBMISSIONTYPE": "13F-NT",
         "REPORTTYPE": "13F NOTICE", "AMENDMENTNO": None, "AMENDMENTTYPE": None},
    ]
    assert rd.resolve_r3(filings, period="2026-03-31",
                         scope_completeness_verified=True) == "N/D"


def test_p66_amendment_chain_hueco():
    '''14.3.2: cadena 1, 3 (falta 2) -> R3 = N/D.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    chain = [
        {"AMENDMENTNO": 1, "AMENDMENTTYPE": "RESTATEMENT"},
        {"AMENDMENTNO": 3, "AMENDMENTTYPE": "NEW_HOLDINGS"},
    ]
    assert rd.validate_amendment_chain(chain) == "N/D"

def test_p66_amendment_chain_duplicado():
    '''14.3.2: cadena 1, 1 (duplicado) -> R3 = N/D.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    chain = [
        {"AMENDMENTNO": 1, "AMENDMENTTYPE": "RESTATEMENT"},
        {"AMENDMENTNO": 1, "AMENDMENTTYPE": "NEW_HOLDINGS"},
    ]
    assert rd.validate_amendment_chain(chain) == "N/D"


def test_p66_amendment_chain_valida():
    '''14.3.2: cadena 1, 2 valida -> OK.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    chain = [
        {"AMENDMENTNO": 1, "AMENDMENTTYPE": "RESTATEMENT"},
        {"AMENDMENTNO": 2, "AMENDMENTTYPE": "NEW_HOLDINGS"},
    ]
    assert rd.validate_amendment_chain(chain) == "OK"


def test_p66_mapping_formnum_cik_1_a_1():
    '''14.3.5: FormNum -> 1 CIK -> IDENTITY_RESOLVED.'''
    import pandas as pd
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    df = pd.DataFrame([
        {"FORM13FFILENUMBER": "028-12345", "CIK": "0000000001",
         "PERIODOFREPORT": "2026-03-31", "ACCESSION_NUMBER": "acc-1"},
        {"FORM13FFILENUMBER": "28-12345",  "CIK": "0000000001",
         "PERIODOFREPORT": "2026-03-31", "ACCESSION_NUMBER": "acc-2"},
    ])
    m = rd.build_formnum_cik_mapping(df, "2026-03-31")
    status, cik = rd.resolve_formnum_to_cik("028-12345", m)
    assert status == "IDENTITY_RESOLVED"
    assert cik == "0000000001"


def test_p66_mapping_formnum_cik_conflict():
    '''14.3.5: FormNum -> >1 CIK -> CONFLICT.'''
    import pandas as pd
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    df = pd.DataFrame([
        {"FORM13FFILENUMBER": "028-12345", "CIK": "0000000001",
         "PERIODOFREPORT": "2026-03-31", "ACCESSION_NUMBER": "acc-1"},
        {"FORM13FFILENUMBER": "028-12345", "CIK": "0000000002",
         "PERIODOFREPORT": "2026-03-31", "ACCESSION_NUMBER": "acc-2"},
    ])
    m = rd.build_formnum_cik_mapping(df, "2026-03-31")
    status, cik = rd.resolve_formnum_to_cik("028-12345", m)
    assert status == "CONFLICT"
    assert cik is None


def test_p66_r4_conflict_prevalece_sobre_match():
    '''14.3.4: fila con CIK=A + FormNum->C -> CONFLICT (no MATCH).

    La implementacion NO puede elegir la fila consistente e ignorar
    la contradictoria respecto de A.
    '''
    import pandas as pd
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    cover = pd.DataFrame([
        {"FORM13FFILENUMBER": "028-11111", "CIK": "A",
         "PERIODOFREPORT": "2026-03-31"},
        {"FORM13FFILENUMBER": "028-22222", "CIK": "C",
         "PERIODOFREPORT": "2026-03-31"},
    ])
    m = rd.build_formnum_cik_mapping(cover, "2026-03-31")
    rows = [
        {"CIK": "A", "FORM13FFILENUMBER": "028-11111"},
        {"CIK": "A", "FORM13FFILENUMBER": "028-22222"},
    ]
    assert rd.resolve_r4("A", rows, m) == "CONFLICT"


def test_p66_r4_conflict_otro_manager_no_contamina():
    '''14.3.4: conflicto de manager C no invalida MATCH de A.'''
    import pandas as pd
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    cover = pd.DataFrame([
        {"FORM13FFILENUMBER": "028-11111", "CIK": "A",
         "PERIODOFREPORT": "2026-03-31"},
        {"FORM13FFILENUMBER": "028-33333", "CIK": "D",
         "PERIODOFREPORT": "2026-03-31"},
    ])
    m = rd.build_formnum_cik_mapping(cover, "2026-03-31")
    rows = [
        {"CIK": "A", "FORM13FFILENUMBER": "028-11111"},
        # Fila C INCONSISTENT: CIK=C y FormNum->D.
        {"CIK": "C", "FORM13FFILENUMBER": "028-33333"},
    ]
    assert rd.resolve_r4("A", rows, m) == "MATCH"


def test_p66_r4_nd_identidad_no_resuelta():
    '''14.3.4: fila con identidad no resuelta -> N/D (no NO_MATCH).'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    rows = [
        # Sin CIK, FormNum que no resuelve.
        {"CIK": None, "FORM13FFILENUMBER": None},
    ]
    assert rd.resolve_r4("A", rows, {}) == "N/D"


def test_p66_formnum_padding_flexible():
    '''14.3.6: 028-694 -> 028-00694; 28-4545 -> 028-04545.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.canonicalize_form13f_filenumber("028-694") == "028-00694"
    assert rd.canonicalize_form13f_filenumber("28-4545") == "028-04545"


def test_p66_formnum_prefijo_invalido():
    '''14.3.6: prefijo != {28, 028} -> UNRESOLVED (None).'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.canonicalize_form13f_filenumber("123-45") is None
    assert rd.canonicalize_form13f_filenumber("02-12345") is None


def test_p66_formnum_sufijo_6_digitos():
    '''14.3.6: 028-064460 -> sufijo 64460 -> 028-64460.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.canonicalize_form13f_filenumber("028-064460") == "028-64460"


def test_p66_formnum_sufijo_7_digitos_unresolved():
    '''14.3.6: 028-2813114 -> UNRESOLVED (no truncar).'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.canonicalize_form13f_filenumber("028-2813114") is None


def test_p66_formnum_none_o_vacio():
    '''14.3.6: None / vacio / sin "-" -> UNRESOLVED.'''
    from src.institutional_accumulation.aggregation import reporting_dedup as rd
    assert rd.canonicalize_form13f_filenumber(None) is None
    assert rd.canonicalize_form13f_filenumber("") is None
    assert rd.canonicalize_form13f_filenumber("028694") is None