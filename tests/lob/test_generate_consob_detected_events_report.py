from __future__ import annotations

import importlib.util
from copy import deepcopy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "generate_consob_detected_events_report.py"
)


def load_module():
    assert SCRIPT_PATH.exists(), "CONSOB detected-event report generator is missing"
    spec = importlib.util.spec_from_file_location(
        "generate_consob_detected_events_report", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(
    timestamp: str,
    *,
    actor_key: str,
    cluster_id: str,
    anchor: str = "passive",
    side: str = "bid",
    quantity: float = 20.0,
    price: float = 100.0,
) -> dict[str, Any]:
    return {
        "cluster_start_ts": timestamp,
        "cluster_end_ts": timestamp,
        "actor_key": actor_key,
        "identity_level": actor_key.split(":", 1)[0],
        "identity_fallback_flag": actor_key.startswith("firm:"),
        "execution_cluster_id": cluster_id,
        "execution_anchor_mode": anchor,
        "execution_side": side,
        "deceptive_side": "ask" if side == "bid" else "bid",
        "execution_quantity": quantity,
        "execution_vwap": price,
        "child_fill_count": 1,
        "execution_price_level_count": 1,
        "candidate_deceptive_order_count_pre": 3,
        "candidate_deceptive_visible_qty_pre": 80.0,
        "candidate_deceptive_min_age_seconds_pre": 1.25,
        "candidate_deceptive_qty_weighted_depth_distance_ticks_pre": 2.5,
        "matched_deceptive_cancel_count_window": 2,
        "matched_deceptive_cancel_visible_qty_window": 60.0,
        "matched_deceptive_cancel_fraction_window": 0.75,
        "matched_deceptive_cancel_min_delay_seconds": 0.01,
        "matched_deceptive_cancel_max_delay_seconds": 0.5,
        "withdrawal_to_execution_ratio": 3.0,
        "weighted_withdrawal_to_execution_ratio": 2.8,
        "favorable_mid_move_pre_fill": 0.2,
        "post_cancel_mid_reversion": 0.1,
        "MSCI_resting_profile": 0.4,
        "withdrawal_profile_scale_event": 1.2,
        "gate_rapid_matched_withdrawal": True,
        "gate_small_fill_relative_to_withdrawal": True,
        "gate_favorable_pre_fill_move": True,
        "gate_cancel_anchored_reversion": True,
        "spoofing_compatible_sequence": True,
    }


def _audit(
    strict_timestamp: str = "2024-06-13T10:00:00",
    *,
    identity_granularity_aligned: bool = False,
    anchor: str = "passive",
    side: str = "bid",
) -> dict[str, Any]:
    event = {
        "event_alias": "detector_event_aaaaaaaaaaaa",
        "cluster_start": strict_timestamp,
        "cluster_end": strict_timestamp,
        "execution_anchor_mode": anchor,
        "execution_side": side,
        "execution_quantity": 20.0,
        "execution_vwap": 100.0,
        "has_matched_withdrawal": True,
        "gate_rapid_matched_withdrawal": True,
        "gate_small_fill_relative_to_withdrawal": True,
        "gate_favorable_pre_fill_move": True,
        "gate_cancel_anchored_reversion": True,
        "strict_detection": True,
    }
    source_row = {
        "period_id": "FERRARI-001",
        "actor_alias": "actor_secret_should_not_render",
        "identity_namespace": "firm",
        "identity_granularity_aligned": identity_granularity_aligned,
        "start": "2024-06-13T09:59:55",
        "end": "2024-06-13T10:00:10",
        "detector_events": [event],
    }
    union_row = {
        "union_period_id": "FERRARI-union-001",
        "actor_alias": "actor_secret_should_not_render",
        "identity_granularity_aligned": identity_granularity_aligned,
        "start": "2024-06-13T09:59:55",
        "end": "2024-06-13T10:00:10",
    }
    return {
        "external_source": {
            "source_kind": "primary_pdf",
            "sha256": "a" * 64,
            "local_path_disclosed": False,
        },
        "overall": {
            "source_external_periods": 2,
            "source_timed_periods": 1,
            "source_date_only_periods": 1,
            "union_timed_periods": 1,
            "union_periods_with_raw_subject_event": 1,
            "union_periods_with_recovered_execution": 1,
            "union_periods_with_matched_withdrawal": 1,
            "union_periods_with_strict_subject_scope_detection": 1,
            "identity_aligned_union_timed_periods": int(identity_granularity_aligned),
            "identity_aligned_union_periods_with_strict_detection": int(
                identity_granularity_aligned
            ),
            "identity_unaligned_union_timed_periods": int(
                not identity_granularity_aligned
            ),
            "identity_unaligned_union_periods_with_strict_subject_scope_detection": int(
                not identity_granularity_aligned
            ),
        },
        "datasets": {
            "FERRARI": {
                "source_period_results": [source_row],
                "union_timed_results": [union_row],
            },
            "RISANAMENTO": {
                "source_period_results": [
                    {
                        "period_id": "RISANAMENTO-001",
                        "scope": "date",
                        "start": "2024-09-10T00:00:00",
                        "event_recall_identifiable": False,
                        "date_level_recovered_execution_clusters": 59,
                        "date_level_clusters_with_matched_withdrawal": 1,
                        "date_level_clusters_with_strict_detection": 0,
                    }
                ],
                "union_timed_results": [],
            },
        },
    }


def _run_info() -> dict[str, Any]:
    return {
        "run_id": "canonical-run-v1",
        "validation_status": "passed",
        "validated_at_utc": "2026-07-30T16:49:30+00:00",
        "audit_sha256": "b" * 64,
        "parameters": {
            "withdrawal_window_seconds": 2.0,
            "reversion_horizon_seconds": 2.0,
        },
        "instruments": {
            "FERRARI": {
                "artifact_sha256": "c" * 64,
                "tick_size": 0.1,
            }
        },
    }


def test_build_report_uses_intermediary_and_new_event_tags():
    module = load_module()
    events = {
        "FERRARI": [
            _event(
                "2024-06-13T10:00:00",
                actor_key="firm:SECRET_FIRM",
                cluster_id="EC-P-1",
            ),
            _event(
                "2024-06-13T10:01:00",
                actor_key="client_original:0",
                cluster_id="EC-P-2",
                quantity=30.0,
                price=101.0,
            ),
        ]
    }

    report = module.build_report(
        events_by_instrument=events,
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "# Relazione sugli eventi compatibili con un possibile schema di spoofing" in report
    assert "**2** sequenze" in report
    assert "| FERRARI-E001 | CONSOB_MATCH_INTERMEDIARIO" in report
    assert "FERRARI-union-001" in report
    assert "| FERRARI-E002 | NUOVO_EVENTO" in report
    assert "| FERRARI | 13/06/2024 | 2 | 2 | 0 | 0 | 1 | 1 | 1 |" in report
    assert "`CONSOB_MATCH_SOGGETTO`" in report
    assert "`CONSOB_MATCH_INTERMEDIARIO`" in report
    assert "`NUOVO_EVENTO`" in report
    assert "CONSOB_MATCH_ESATTO" not in report
    assert "Nessun riscontro CONSOB" not in report
    assert "Tutti i criteri tra quelli con identificazione allineata" in report
    assert "### Lettura operativa per la prima istruttoria" in report
    assert "| Registro ristretto | 2 sequenze complete |" in report
    assert (
        "| Qualità dell'attribuzione | 0/2 (0,0%) con codice cliente; "
        "1/2 (50,0%) solo tramite intermediario; 1/2 (50,0%) non attribuibili |"
    ) in report
    assert "| Confronto intraday CONSOB | 1/1 (100,0%) intervalli con una sequenza completa |" in report
    assert "corrispondenza a orologio registrato" in report
    assert "**Passi istruttori suggeriti.**" in report
    assert "### Schede dei riscontri a livello di intermediario" in report
    assert "| Riferimento: `FERRARI-E001` | Intervallo: FERRARI-union-001" in report
    assert "Confermare base temporale e sincronizzazione" in report
    assert "Con soggetto identificato" not in report
    assert "Con identità esatta" not in report
    assert "Intermediario: `SECRET_FIRM`" in report
    assert "Codice cliente non disponibile per l'evento" in report
    assert "Identificativo non attribuibile: `0`" in report
    assert "il valore tecnico non identifica un cliente o un intermediario" in report
    assert "FERRARI-S001" not in report
    assert "L'operazione comprende una sola esecuzione, a un unico prezzo e dura 0,0 ms" in report
    assert (
        "Prima dell'esecuzione erano visibili 3 ordini di vendita, per un totale "
        "di 80 unità."
    ) in report
    assert (
        "Al momento dell'esecuzione, il più recente era nel libro degli ordini da "
        "1,250 s; la distanza media, ponderata per quantità, dalla migliore proposta "
        "di vendita era di 2,50 tick."
    ) in report
    assert (
        "Dopo l'esecuzione sono state rilevate, tra gli ordini considerati, 2 "
        "cancellazioni, per un totale di 60 unità."
    ) in report
    assert (
        "La quantità cancellata corrisponde al 75,0% della quantità visibile prima "
        "dell'esecuzione."
    ) in report
    assert "Prima dell'acquisto, il midprice è sceso di 0,20" in report
    assert report.count("| FERRARI-E00") == 2
    assert "firm:SECRET_FIRM" not in report
    assert "actor_secret_should_not_render" not in report
    assert "EC-P-1" not in report
    assert "/home/" not in report
    assert "non dimostra né un intento manipolativo né una violazione" in report
    assert "Riservato — contiene identificativi originali di clienti o intermediari" in report
    assert "dati pseudonimizzati" not in report
    assert "orari così come sono registrati nei due archivi" in report
    assert "verificare fuso orario" in report
    assert "### Condizioni di ingresso nel registro" in report
    assert "1. **Ordini di segno opposto:**" in report
    assert "2. **Ritiro successivo:**" in report
    assert "3. **Dimensione del ritiro:**" in report
    assert "4. **Movimento precedente all'esecuzione:**" in report
    assert "5. **Movimento successivo alle cancellazioni:**" in report
    assert (
        "Le cinque condizioni sono cumulative e volutamente restrittive come filtro operativo"
        in report
    )
    assert "non costituiscono requisiti giuridici necessari" in report
    assert (
        "Non è consentito effettuare manipolazioni di mercato o tentare di effettuare "
        "manipolazioni di mercato"
    ) in report
    assert "Il mero intento, non accompagnato da condotte osservabili" in report
    assert "[1] https://eur-lex.europa.eu/legal-content/IT/TXT/?uri=CELEX:32014R0596" in report
    assert "### Cosa coincide, e a quale livello" in report
    assert "**Presenza di attività:** in 1 intervallo su 1" in report
    assert "**Sequenza comportamentale:** in 1 intervallo su 1" in report
    assert "**Identificazione:** in 0 intervalli su 1" in report
    assert "non è una conferma indipendente dell'evento segnalato da CONSOB" in report
    assert "Tutti i criteri tra quelli con identificazione allineata" in report
    assert "ordini di segno opposto all'operazione eseguita" in report
    assert "le cancellazioni ricevono un peso progressivamente minore" in report
    assert "soltanto quando l'intero intervallo è osservabile" in report
    assert "Non è imposto un tempo minimo di permanenza" in report
    assert "individuata senza ambiguità" in report
    assert "In caso di ambiguità, il riscontro non è riportato nel registro." in report
    assert "la relazione non viene prodotta" not in report
    assert "Il riscontro riportato nel registro è classificato a livello di intermediario" in report
    assert "I quattro riscontri" not in report
    assert "lo stesso cluster del registro" not in report


@pytest.mark.parametrize(
    "actor_key",
    ["client_original:0.0", "client_original:0.00", "client_original:0e0", "client_original:-0.0"],
)
def test_report_treats_legacy_zero_client_actor_variants_as_unattributable(actor_key):
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key=actor_key,
        cluster_id="EC-Z-1",
    )

    report = module.build_report(
        events_by_instrument={"FERRARI": [event]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "Identificativo non attribuibile" in report
    assert "In 1 eventi l'unico valore identificativo disponibile" in report
    assert "| FERRARI | 13/06/2024 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |" in report


def test_build_report_uses_subject_tag_for_identity_aligned_match():
    module = load_module()
    report = module.build_report(
        events_by_instrument={
            "FERRARI": [
                _event(
                    "2024-06-13T10:00:00",
                    actor_key="client_original:CLIENT_123",
                    cluster_id="EC-P-1",
                )
            ]
        },
        audit=_audit(identity_granularity_aligned=True),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "| FERRARI-E001 | CONSOB_MATCH_SOGGETTO" in report
    assert "CONSOB_MATCH_INTERMEDIARIO<br>" not in report
    assert "Il riscontro classificato a livello del soggetto indica" in report
    assert "classificato a livello di intermediario: coincidono" not in report


def test_build_report_displays_the_original_client_identifier():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="client_original:CLIENT_123",
        cluster_id="EC-P-1",
    )

    report = module.build_report(
        events_by_instrument={"FERRARI": [event]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "Cliente: `CLIENT_123`" in report
    assert "client_original:CLIENT_123" not in report


def test_build_report_uses_natural_wording_for_one_cancellation():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    event["matched_deceptive_cancel_count_window"] = 1
    event["matched_deceptive_cancel_min_delay_seconds"] = 0.01
    event["matched_deceptive_cancel_max_delay_seconds"] = 0.01

    report = module.build_report(
        events_by_instrument={"FERRARI": [event]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "La cancellazione avviene 10,0 ms dopo l'esecuzione." in report
    assert "Il ritardo va da 10,0 a 10,0 ms" not in report


def test_build_report_orders_events_by_descending_wmsci_without_renumbering_ids():
    module = load_module()
    earlier = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    later = _event(
        "2024-06-13T10:01:00",
        actor_key="client_original:0",
        cluster_id="EC-P-2",
    )
    earlier["withdrawal_profile_scale_event"] = 1.25
    later["withdrawal_profile_scale_event"] = 4.75
    nexi = _event(
        "2024-04-23T09:00:00",
        actor_key="firm:NEXI_FIRM",
        cluster_id="EC-P-3",
        price=5.5,
    )
    nexi["withdrawal_profile_scale_event"] = 9.5
    run_info = _run_info()
    run_info["instruments"]["NEXI"] = {
        "artifact_sha256": "d" * 64,
        "tick_size": 0.002,
    }

    report = module.build_report(
        events_by_instrument={"FERRARI": [earlier, later], "NEXI": [nexi]},
        audit=_audit(),
        run_info=run_info,
        report_date="6 agosto 2026",
    )

    assert (
        report.index("| NEXI-E001 |")
        < report.index("| FERRARI-E002 |")
        < report.index("| FERRARI-E001 |")
    )
    assert "Gli eventi sono ordinati per WMSCI, dal valore più alto al più basso." in report


def test_build_report_is_stable_when_complete_sort_keys_tie():
    module = load_module()
    actor_a = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:ACTOR_A",
        cluster_id="EC-P-TIED",
        price=100.0,
    )
    actor_b = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:ACTOR_B",
        cluster_id="EC-P-TIED",
        price=101.0,
    )

    forward = module.build_report(
        events_by_instrument={"FERRARI": [actor_a, actor_b]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )
    reversed_input = module.build_report(
        events_by_instrument={"FERRARI": [actor_b, actor_a]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert reversed_input == forward
    assert "| FERRARI-E001 | CONSOB_MATCH_INTERMEDIARIO" in forward


def test_build_report_uses_plain_italian_instead_of_internal_pipeline_jargon():
    module = load_module()
    report = module.build_report(
        events_by_instrument={
            "FERRARI": [
                _event(
                    "2024-06-13T10:00:00",
                    actor_key="firm:SECRET_FIRM",
                    cluster_id="EC-P-1",
                )
            ]
        },
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "Il registro raccoglie" in report
    assert "soddisfano tutti i criteri di selezione descritti di seguito" in report
    assert "Prima dell'esecuzione" in report
    assert "Codice cliente non disponibile per l'evento" in report
    assert (
        "Acquisto eseguito mediante un ordine di acquisto già presente nel libro degli ordini"
        in report
    )
    assert "L'operazione comprende una sola esecuzione, a un unico prezzo e dura 0,0 ms" in report
    assert "ordini di vendita" in report
    assert "migliore proposta di vendita" in report
    assert "Prima dell'acquisto, il midprice è sceso" in report
    assert "dopo le cancellazioni è risalito in media" in report
    assert (
        "La quantità cancellata corrisponde al 75,0% della quantità visibile prima "
        "dell'esecuzione."
    ) in report
    assert "Il rapporto tra quantità cancellata ed eseguita è 3,00" in report
    assert "attribuendo un peso minore alle cancellazioni più tardive, quello ponderato è 2,80" in report
    assert "lato ask" not in report
    assert "esecuzioni parziali" not in report
    assert "stesso livello di dettaglio identificativo" in report
    assert "scala di 10 secondi" in report
    assert "presenti nel libro degli ordini da non più di 90 s" in report
    assert "### Come leggere WMSCI e MSCI" in report
    assert "In questa relazione WMSCI è la metrica principale" in report
    assert (
        "WMSCI = log(1 + quantità visibile / quantità eseguita) × "
        "log(1 + quantità ritirata ponderata / quantità eseguita) × quota ritirata"
        in report
    )
    assert "La WMSCI è sempre non negativa, non ha un massimo prefissato" in report
    assert "non è una probabilità né una percentuale" in report
    assert "MSCI = SCI / 2 + C_opposta - C_stessa" in report
    assert "è compreso tra -1 e 2" in report
    assert "Un valore vicino a zero non implica necessariamente assenza di cambiamenti" in report
    assert "Nessuno dei due comprende il movimento del prezzo" in report
    assert "La WMSCI determina soltanto l'ordine di presentazione" in report
    assert "non è prevista una soglia minima di WMSCI o MSCI" in report
    for jargon in (
        "detector",
        "gate",
        "pre-fill",
        "fallback firm",
        "(recall)",
        "resting",
        "La riga interessata",
        "lato denaro",
        "lato lettera",
        "sul lato",
        "denaro-lettera",
        "Esecuzione passiva",
        "Esecuzione aggressiva",
        "ancoraggio passivo",
        "ancoraggio aggressivo",
        "modalità aggressiva",
        "direzione favorevole al lato",
        "valore tecnico sentinella",
        "granularità dell'identità",
        "chiave interna",
        "profilo considerato",
        "libro ordini",
    ):
        assert jargon not in report


def test_build_report_describes_aggressive_sale_in_natural_italian():
    module = load_module()
    report = module.build_report(
        events_by_instrument={
            "FERRARI": [
                _event(
                    "2024-06-13T10:00:00",
                    actor_key="firm:SECRET_FIRM",
                    cluster_id="EC-A-1",
                    anchor="aggressive",
                    side="ask",
                )
            ]
        },
        audit=_audit(anchor="aggressive", side="ask"),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "Vendita contro ordini di acquisto già presenti nel libro degli ordini" in report
    assert "Prima dell'esecuzione erano visibili 3 ordini di acquisto" in report
    assert "migliore proposta di acquisto" in report
    assert "Prima della vendita, il midprice è salito" in report
    assert "dopo le cancellazioni è sceso in media" in report
    assert "In 0 eventi" not in report
    assert "Questi casi restano nel registro" not in report
    assert "Gli eventi per i quali l'unico valore identificativo disponibile" not in report


def test_build_report_rounds_price_movements_to_two_decimal_ticks():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    event["post_cancel_mid_reversion"] = 0.0970004

    report = module.build_report(
        events_by_instrument={"FERRARI": [event]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "(0,97 tick)" in report
    assert "0,970004 tick" not in report


def test_it_compact_number_omits_an_empty_decimal_separator():
    module = load_module()

    assert module._it_compact_number(90.0, min_decimals=0) == "90"


def test_build_report_rejects_audit_event_without_one_canonical_match():
    module = load_module()
    events = {
        "FERRARI": [
            _event(
                "2024-06-13T10:01:00",
                actor_key="firm:SECRET_FIRM",
                cluster_id="EC-P-1",
            )
        ]
    }

    with pytest.raises(ValueError, match="exactly one canonical strict event"):
        module.build_report(
            events_by_instrument=events,
            audit=_audit(strict_timestamp="2024-06-13T10:00:00"),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_build_report_matches_native_datetime_values_loaded_from_parquet():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    event["cluster_start_ts"] = datetime.fromisoformat("2024-06-13T10:00:00")
    event["cluster_end_ts"] = datetime.fromisoformat("2024-06-13T10:00:00")

    report = module.build_report(
        events_by_instrument={"FERRARI": [event]},
        audit=_audit(),
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "| FERRARI-E001 | CONSOB_MATCH_INTERMEDIARIO" in report


def test_build_report_uses_configured_time_windows_in_explanatory_prose():
    module = load_module()
    run_info = _run_info()
    run_info["parameters"]["withdrawal_window_seconds"] = 3.5
    run_info["parameters"]["reversion_horizon_seconds"] = 4.0

    report = module.build_report(
        events_by_instrument={
            "FERRARI": [
                _event(
                    "2024-06-13T10:00:00",
                    actor_key="firm:SECRET_FIRM",
                    cluster_id="EC-P-1",
                )
            ]
        },
        audit=_audit(),
        run_info=run_info,
        report_date="6 agosto 2026",
    )

    assert "viene cancellata entro 3,5 secondi" in report
    assert "nell'intervallo di 4 secondi successivo a ciascuna cancellazione" in report
    assert "viene cancellata entro 2 secondi" not in report


def test_build_report_derives_date_only_alert_note_from_audit_values():
    module = load_module()
    audit = _audit()
    date_only = audit["datasets"]["RISANAMENTO"]["source_period_results"][0]
    date_only.update(
        {
            "start": "2025-01-02T00:00:00",
            "date_level_recovered_execution_clusters": 7,
            "date_level_clusters_with_matched_withdrawal": 3,
            "date_level_clusters_with_strict_detection": 2,
        }
    )

    report = module.build_report(
        events_by_instrument={
            "FERRARI": [
                _event(
                    "2024-06-13T10:00:00",
                    actor_key="firm:SECRET_FIRM",
                    cluster_id="EC-P-1",
                )
            ]
        },
        audit=audit,
        run_info=_run_info(),
        report_date="6 agosto 2026",
    )

    assert "La segnalazione RISANAMENTO del 02/01/2025 riporta soltanto la data." in report
    assert "si osservano 7 gruppi di esecuzioni" in report
    assert "3 presentano cancellazioni successive" in report
    assert "2 soddisfano tutti i criteri di selezione" in report
    assert "59 gruppi di esecuzioni" not in report


def test_build_report_rejects_rows_that_do_not_pass_every_strict_gate():
    module = load_module()
    invalid = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    invalid["gate_cancel_anchored_reversion"] = False

    with pytest.raises(ValueError, match="strict-event register contains a failed gate"):
        module.build_report(
            events_by_instrument={"FERRARI": [invalid]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_build_report_rejects_markdown_injection_in_report_date():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )

    with pytest.raises(ValueError, match="report_date"):
        module.build_report(
            events_by_instrument={"FERRARI": [event]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026\n## Sezione non autorizzata",
        )


def test_build_report_rejects_an_impossible_calendar_date():
    module = load_module()
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )

    with pytest.raises(ValueError, match="report_date"):
        module.build_report(
            events_by_instrument={"FERRARI": [event]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="31 febbraio 2026",
        )


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    [
        ("source_period_results", "period_id", "FERRARI-001<br>TESTO-NON-AUTORIZZATO"),
        ("union_timed_results", "union_period_id", "FERRARI-union-001|colonna"),
    ],
)
def test_build_report_rejects_unsafe_audit_period_identifiers(
    collection: str, field: str, value: str
):
    module = load_module()
    audit = _audit()
    audit["datasets"]["FERRARI"][collection][0][field] = value
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )

    with pytest.raises(ValueError, match="period identifier"):
        module.build_report(
            events_by_instrument={"FERRARI": [event]},
            audit=audit,
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("deceptive_side", "bid", "opposite"),
        ("identity_fallback_flag", False, "identity fallback"),
        ("child_fill_count", 0, "positive integer"),
        ("matched_deceptive_cancel_fraction_window", 1.1, "cancellation fraction"),
        ("matched_deceptive_cancel_min_delay_seconds", -0.1, "cancellation delays"),
        ("matched_deceptive_cancel_max_delay_seconds", 2.1, "withdrawal window"),
    ],
)
def test_build_report_rejects_semantically_inconsistent_strict_rows(
    field: str, value: object, message: str
):
    module = load_module()
    invalid = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )
    invalid[field] = value

    with pytest.raises(ValueError, match=message):
        module.build_report(
            events_by_instrument={"FERRARI": [invalid]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_build_report_rejects_unknown_identity_levels_before_rendering():
    module = load_module()
    invalid = _event(
        "2024-06-13T10:00:00",
        actor_key="external_label<br>PRIVATE-SUBJECT:123",
        cluster_id="EC-P-1",
    )
    invalid["identity_level"] = "external_label<br>PRIVATE-SUBJECT"
    invalid["identity_fallback_flag"] = False

    with pytest.raises(ValueError, match="unsupported identity level"):
        module.build_report(
            events_by_instrument={"FERRARI": [invalid]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_build_report_rejects_markup_in_an_original_actor_identifier():
    module = load_module()
    invalid = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET<br>FIRM",
        cluster_id="EC-P-1",
    )

    with pytest.raises(ValueError, match="original actor identifier"):
        module.build_report(
            events_by_instrument={"FERRARI": [invalid]},
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_rendered_event_row_validator_rejects_a_missing_event():
    module = load_module()

    with pytest.raises(ValueError, match="event-row IDs"):
        module._validate_rendered_event_rows(
            ["| FERRARI-E001 | a | b | c | d | e | f |"],
            ["FERRARI-E001", "FERRARI-E002"],
        )


def test_rendered_event_row_validator_rejects_wrong_wmsci_order():
    module = load_module()
    rows = [
        "| FERRARI-E002 | a | b | c | d | e | f |",
        "| FERRARI-E001 | a | b | c | d | e | f |",
    ]

    with pytest.raises(ValueError, match="order"):
        module._validate_rendered_event_rows(
            rows,
            ["FERRARI-E001", "FERRARI-E002"],
        )


def test_rendered_event_row_validator_accepts_an_escaped_data_pipe():
    module = load_module()
    module._validate_rendered_event_rows(
        [
            "| FERRARI-E001 | CONSOB_MATCH_SOGGETTO<br>fonte FERRARI\\|005 "
            "| b | c | d | e | f |"
        ],
        ["FERRARI-E001"],
    )


@pytest.mark.parametrize(
    "row",
    [
        "| FERRARI-E001 | a | b | c | d | e |",
        "| FERRARI-E001 | a | b | c | d | e | f | g |",
    ],
)
def test_rendered_event_row_validator_rejects_non_seven_cell_rows(row: str):
    module = load_module()

    with pytest.raises(ValueError, match="seven Markdown cells"):
        module._validate_rendered_event_rows([row], ["FERRARI-E001"])


def test_build_report_fails_closed_for_structurally_ambiguous_audit_event():
    module = load_module()
    events = {
        "FERRARI": [
            _event(
                "2024-06-13T10:00:00",
                actor_key="firm:FIRM_A",
                cluster_id="EC-P-1",
            ),
            _event(
                "2024-06-13T10:00:00",
                actor_key="firm:FIRM_B",
                cluster_id="EC-P-2",
            ),
        ]
    }

    with pytest.raises(ValueError, match="found 2"):
        module.build_report(
            events_by_instrument=events,
            audit=_audit(),
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_build_report_rejects_one_audit_event_assigned_to_two_subject_aliases():
    module = load_module()
    audit = _audit()
    duplicate = deepcopy(audit["datasets"]["FERRARI"]["source_period_results"][0])
    duplicate["period_id"] = "FERRARI-002"
    duplicate["actor_alias"] = "actor_different_subject"
    audit["datasets"]["FERRARI"]["source_period_results"].append(duplicate)
    event = _event(
        "2024-06-13T10:00:00",
        actor_key="firm:SECRET_FIRM",
        cluster_id="EC-P-1",
    )

    with pytest.raises(ValueError, match="inconsistent duplicate audit event"):
        module.build_report(
            events_by_instrument={"FERRARI": [event]},
            audit=audit,
            run_info=_run_info(),
            report_date="6 agosto 2026",
        )


def test_canonical_report_snapshot_when_inputs_are_available():
    module = load_module()
    root = SCRIPT_PATH.parents[1]
    run_root = (
        root
        / "outputs/spoofing_metrics/20260730_160131_all_instruments_json_empirical_top10_withdrawal_fix"
    )
    audit_path = (
        root
        / "outputs/consob_alert_comparison/20260731_140816_source_alert_event_mapping_v1"
        / "external_alert_detector_overlap_audit.json"
    )
    report_path = root / "reports/consob/20260806/report_eventi_rilevati.md"
    if not run_root.is_dir() or not audit_path.is_file() or not report_path.is_file():
        pytest.skip("canonical private CONSOB artifacts are not available")

    events, audit, run_info = module.load_canonical_inputs(run_root, audit_path)
    report = module.build_report(
        events_by_instrument=events,
        audit=audit,
        run_info=run_info,
        report_date="6 agosto 2026",
    )
    rows = re.findall(
        r"^\| ((?:FERRARI|NEXI|RISANAMENTO)-E\d{3}) \| (.*?) \|",
        report,
        flags=re.MULTILINE,
    )
    counts = {
        instrument: sum(event_id.startswith(f"{instrument}-") for event_id, _ in rows)
        for instrument in ("FERRARI", "NEXI", "RISANAMENTO")
    }
    tagged = {
        event_id: "<br>".join(context.split("<br>")[:3])
        for event_id, context in rows
        if context.startswith("CONSOB_MATCH_")
    }

    assert counts == {"FERRARI": 378, "NEXI": 231, "RISANAMENTO": 12}
    assert tagged == {
        "FERRARI-E061": "CONSOB_MATCH_INTERMEDIARIO<br>Intervallo: FERRARI-union-002<br>Fonte: FERRARI-005",
        "FERRARI-E102": "CONSOB_MATCH_INTERMEDIARIO<br>Intervallo: FERRARI-union-009<br>Fonte: FERRARI-022",
        "FERRARI-E199": "CONSOB_MATCH_INTERMEDIARIO<br>Intervallo: FERRARI-union-012<br>Fonte: FERRARI-016",
        "FERRARI-E330": "CONSOB_MATCH_INTERMEDIARIO<br>Intervallo: FERRARI-union-025<br>Fonte: FERRARI-010",
    }
    assert "**Presenza di attività:** in 34 intervalli su 34" in report
    assert "**Sequenza comportamentale:** in 30 intervalli su 34" in report
    assert "In 4 intervalli su 34 compare almeno una sequenza che soddisfa tutti i criteri" in report
    assert "**Identificazione:** in 4 intervalli su 34" in report
    assert "nessuno dei 4 contiene una sequenza che soddisfa tutti i criteri" in report
    assert (
        "I 4 riscontri completi appartengono invece agli altri 30 intervalli, nei quali "
        "il confronto è possibile soltanto a un livello più ampio"
    ) in report
    assert (
        "Il risultato finale è quindi: **4 sovrapposizioni temporali e comportamentali "
        "a livello di intermediario, ma 0 corrispondenze complete con identificazione allineata**."
    ) in report
    assert "| FERRARI | 35 | 0 | 33 | 33/33 | 30/33 | 4/33 | 0/3 |" in report
    assert "| NEXI | 1 | 0 | 1 | 1/1 | 0/1 | 0/1 | 0/1 |" in report
    assert "| RISANAMENTO | 1 | 1 | — | — | — | — | — |" in report
    assert report == report_path.read_text(encoding="utf-8")
