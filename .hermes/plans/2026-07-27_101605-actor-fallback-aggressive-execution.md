# Actor fallback e ramo aggressive-execution Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Estendere il detector affinché (1) usi l’original-client shortcode quando disponibile e ripieghi esplicitamente su `FIRMID` quando manca, e (2) analizzi in un ramo separato anche le esecuzioni aggressive, senza mescolare granularità d’identità o popolazioni con semantiche diverse.

**Architecture:** Introdurre un contratto di identità centralizzato e namespaced (`actor_key`) derivato da `NMSC_ORIGINALCLIENTIDSHORTCODE` oppure, solo come fallback, da `FIRMID`. Generalizzare clustering, stato, cancellazioni e aggregazioni dall’attuale `client_id` a `actor_key`, mantenendo gli identificativi grezzi per audit. Il ramo `aggressive` condividerà attribuzione delle cancellazioni e gate comportamentali con il ramo `passive`, ma conserverà `execution_anchor_mode` in ogni artefatto e verrà aggregato separatamente per evitare soglie o denominatori impropriamente pooled.

**Tech Stack:** Python 3, Polars, dataclass e funzioni pure, CLI `argparse`, configurazione JSON, Parquet/CSV/JSON, pytest con fixture sintetiche deterministiche.

---

## 1. Contesto corrente e vincoli scientifici

- La ricostruzione LOB conserva già separatamente `firm_id` e `client_original_id` in `src/spoofing_detection/lob/normalize.py`, `models.py` e `panel.py`.
- `src/spoofing_detection/lob/spoofing_metrics.py:383-446` esclude oggi:
  - ogni fill senza `client_original_id` con `missing_client_id`;
  - ogni fill con `AGGRESSIVEORDER=Y` con `aggressive_execution`;
  - le esecuzioni aggressive non-limit e prive di ordine attivo prima del fill.
- `src/spoofing_detection/lob/execution_clusters.py` clusterizza soltanto fill passivi e usa `(partition, order, client, side, price)` come identità.
- Stato, MCPS, production readiness, dashboard e dossier usano `client_id` come chiave canonica.
- Gli alert esterni hanno mostrato due classi di mancata copertura:
  - Ferrari: `NMSC_ORIGINALCLIENTIDSHORTCODE` assente, `FIRMID` disponibile;
  - Nexi: attore identificato, ma fill aggressivi esclusi.
- Gli alert esterni sono benchmark di copertura del generatore di candidati, non ground truth di manipolazione.
- Non si devono modificare le soglie MSCI/WMSCI o i quattro gate stretti per “far passare” questi casi. Prima va corretta la popolazione ammissibile.
- Il fallback firm-level aggrega potenzialmente più clienti dello stesso intermediario. Deve quindi rimanere riconoscibile e non essere presentato come client-level.
- I punteggi `passive` e `aggressive` non vanno pooled nel primo rilascio: le popolazioni e la semantica dell’esecuzione sono diverse e le soglie correnti sono state osservate sul ramo passivo.

## 2. Contratto dati proposto

Ogni artefatto analitico event-level, cluster-level e alert-level deve includere:

| Colonna | Semantica |
|---|---|
| `actor_key` | Chiave canonica namespaced: `client_original:<id>` o `firm:<id>`. È la chiave di join/grouping. |
| `actor_id` | Valore grezzo selezionato, senza namespace. |
| `identity_level` | `client_original` oppure `firm`. |
| `identity_source` | Campo sorgente: `NMSC_ORIGINALCLIENTIDSHORTCODE` oppure `FIRMID`. |
| `identity_fallback_flag` | `true` soltanto per il fallback firm-level. |
| `event_client_original_id` | Identificativo client originale grezzo, nullable. |
| `event_firm_id` | Identificativo firm grezzo, nullable. |
| `execution_anchor_mode` | `passive` oppure `aggressive`. |
| `execution_price_source` | `LASTTRADEDPX` o, solo nel ramo passivo e se documentato, `active_order_price`. |

Regole invarianti:

1. Se il client è valorizzato, scegliere sempre `client_original`, anche se è presente `FIRMID`.
2. Usare `FIRMID` solo quando il client è veramente mancante/vuoto.
3. Non usare una chiave composita firm+client e non confrontare `actor_id` senza `identity_level`.
4. Se entrambi mancano, rifiutare con `missing_actor_identity`.
5. Non copiare un `FIRMID` nella colonna legacy `client_id`; quest’ultima deve restare nullable e rappresentare soltanto un client reale durante la migrazione.
6. Un’esecuzione client-level non può attribuirsi ordini firm-level e viceversa, anche se condividono `FIRMID`.
7. Aggregare score/alert almeno per `(actor_key, execution_anchor_mode)`; `identity_level` deve essere univoco per `actor_key`.

## 3. Semantica del nuovo ramo aggressive

Classificazione role-level proposta:

- `passive`: `PASSIVEORDER=Y` e `AGGRESSIVEORDER!=Y`;
- `aggressive`: `AGGRESSIVEORDER=Y` e `PASSIVEORDER!=Y`;
- entrambi `Y`: rifiuto auditabile `ambiguous_execution_role`;
- nessuno dei due: rifiuto auditabile `missing_execution_role` o mantenimento fuori dalla popolazione, secondo la semantica già osservata nei test.

Ammissibilità aggressive:

- evento `fill`;
- timestamp, side e attore validi;
- `LASTSHARES` finito e strettamente positivo;
- `LASTTRADEDPX` finito e strettamente positivo;
- nessun requisito di ordine attivo prima del fill;
- market order e crossing limit sono ammessi;
- l’ordine aggressivo non viene trattato come liquidità resting.

Semantica microstrutturale:

- `execution_side` resta la side dell’ordine aggressivo;
- `deceptive_side = opposite(execution_side)` come nel ramo passivo;
- il profilo candidato è lo stato dello stesso `actor_key` sul lato `deceptive_side` immediatamente prima del primo fill del cluster;
- i fill aggressivi dello stesso ordine/sweep possono avere prezzi diversi: clusterizzare per `(partition, actor_key, order_id, side, execution_anchor_mode)` e gap temporale, non per prezzo; `event_price` cluster-level è il VWAP dei child fill;
- i rapporti “same-level client visible quantity” nati per l’ordine passivo non vanno reinterpretati artificialmente. Nel ramo aggressive vanno emessi come `null` o sostituiti con diagnostiche chiaramente rinominate; il gate `fill_qty < withdrawn_qty` resta invece valido e branch-neutral;
- ogni cluster id deve includere il ramo, per esempio `EC-P-...` / `EC-A-...`, così la provenance non è ambigua.

## 4. Criteri di accettazione

1. Un fill con client presente continua a produrre `actor_key=client_original:<id>` e risultati passivi numericamente invariati, salvo le nuove colonne/ID cluster documentate.
2. Un fill con client mancante e firm presente non viene più rifiutato come `missing_client_id`; viene processato come `identity_level=firm`.
3. Due valori grezzi uguali, uno client e uno firm, non collidono.
4. Candidati, cancellazioni, stato e cluster sono collegati soltanto quando condividono lo stesso `actor_key`.
5. Un fill aggressive valido entra in un cluster `execution_anchor_mode=aggressive` usando `LASTTRADEDPX` e `LASTSHARES`.
6. Un market order aggressive valido non richiede un `ActiveOrder` preesistente.
7. Un fill con flag passivo/aggressivo ambiguo rimane nei rejected outputs con ragione esplicita.
8. MCPS, feature di sessione e alert sono separati per ramo; nessun alert unisce eventi passive e aggressive nello stesso denominatore.
9. Gli artefatti e i metadata dichiarano schema/identity/branch; output legacy non vengono riutilizzati automaticamente.
10. Nel rerun reale:
    - le 35 finestre Ferrari non risultano più perse per `missing_client_id` quando `FIRMID` è disponibile;
    - la finestra Nexi contiene almeno un cluster aggressive ammissibile anziché quattro righe escluse come `aggressive_execution`;
    - non è richiesto che tali cluster superino i gate stretti: quello è un risultato empirico da riportare, non un criterio da forzare.

---

### Task 1: Introdurre il resolver canonico dell’attore

**Objective:** Centralizzare selezione, normalizzazione e matching dell’identità senza duplicare logica in metriche, cluster e cancellazioni.

**Files:**
- Create: `src/spoofing_detection/lob/actor_identity.py`
- Create: `tests/lob/test_actor_identity.py`
- Modify: `src/spoofing_detection/lob/models.py:9-25` solo se serve un helper tipizzato per `ActiveOrder`

**Step 1: Scrivere i test RED**

Coprire almeno:

```python
from spoofing_detection.lob.actor_identity import resolve_actor_identity


def test_client_identity_has_priority_over_firm():
    actor = resolve_actor_identity(client_original_id="31425", firm_id="130358")
    assert actor.actor_key == "client_original:31425"
    assert actor.actor_id == "31425"
    assert actor.identity_level == "client_original"
    assert actor.identity_fallback_flag is False


def test_missing_client_falls_back_to_firm_without_calling_it_client():
    actor = resolve_actor_identity(client_original_id=None, firm_id="157922_3")
    assert actor.actor_key == "firm:157922_3"
    assert actor.actor_id == "157922_3"
    assert actor.identity_level == "firm"
    assert actor.identity_fallback_flag is True


def test_namespaces_prevent_raw_identifier_collision():
    client = resolve_actor_identity(client_original_id="123", firm_id="F")
    firm = resolve_actor_identity(client_original_id=None, firm_id="123")
    assert client.actor_key != firm.actor_key


def test_missing_both_identities_returns_none():
    assert resolve_actor_identity(client_original_id=" ", firm_id=None) is None
```

Aggiungere casi per stringhe `"null"`, `"none"`, `"nan"`, tipi numerici convertiti senza `.0` spurio se la normalizzazione corrente lo consente, e matching fra evento e `ActiveOrder`.

**Step 2: Verificare il fallimento**

Run:

```bash
pytest tests/lob/test_actor_identity.py -q
```

Expected: FAIL per modulo/funzione non ancora esistente.

**Step 3: Implementare il resolver minimo**

Interfaccia proposta:

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActorIdentity:
    actor_key: str
    actor_id: str
    identity_level: str
    identity_source: str
    identity_fallback_flag: bool


def resolve_actor_identity(*, client_original_id: Any, firm_id: Any) -> ActorIdentity | None:
    client = normalize_identity_value(client_original_id)
    if client is not None:
        return ActorIdentity(
            actor_key=f"client_original:{client}",
            actor_id=client,
            identity_level="client_original",
            identity_source="NMSC_ORIGINALCLIENTIDSHORTCODE",
            identity_fallback_flag=False,
        )
    firm = normalize_identity_value(firm_id)
    if firm is not None:
        return ActorIdentity(
            actor_key=f"firm:{firm}",
            actor_id=firm,
            identity_level="firm",
            identity_source="FIRMID",
            identity_fallback_flag=True,
        )
    return None
```

Aggiungere funzioni pure `actor_identity_from_event(...)`, `actor_identity_from_order(...)` e `same_actor(...)`; non spargere confronti client/firm condizionali nel resto del codice.

**Step 4: Verificare GREEN**

Run:

```bash
pytest tests/lob/test_actor_identity.py -q
```

Expected: tutti i test PASS.

**Step 5: Review locale**

Controllare che la chiave non contenga dati aggiuntivi, che `firm` non venga mai etichettato `client_original`, e che non sia stato introdotto un fallback a `MSC_EVENTCLIENTIDSHORTCODE` non richiesto.

---

### Task 2: Rendere actor-aware esposizioni top-N e stato temporale

**Objective:** Calcolare DWI/MSCI sul profilo dell’attore risolto, inclusi ordini firm-level quando il client manca.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:208-361`
- Modify: `tests/lob/test_spoofing_metrics.py:194-303`

**Step 1: Scrivere test RED per esposizioni firm fallback**

Aggiungere fixture con:

- ordine `client=C1, firm=F1` → profilo `client_original:C1`;
- ordine `client=None, firm=F2` → profilo `firm:F2`;
- client raw `F2` e firm raw `F2` → due profili distinti;
- richiesta esplicita di zero-profile per un `actor_key` senza ordini top-N.

Assertions essenziali:

```python
assert set(rows_df["actor_key"]) == {"client_original:C1", "firm:F2"}
assert rows_df.filter(pl.col("actor_key") == "firm:F2")["identity_level"][0] == "firm"
assert "client_id" not in actor_grouping_columns
```

**Step 2: Verificare RED**

Run:

```bash
pytest tests/lob/test_spoofing_metrics.py -q -k "actor or top_n_exposures"
```

Expected: FAIL perché la funzione usa ancora solo `order.client_original_id`.

**Step 3: Generalizzare la funzione**

- Rinominare internamente `compute_client_top_n_exposures` in `compute_actor_top_n_exposures`.
- Sostituire `client_ids`/`include_zero_client_ids` con `actor_keys`/`include_zero_actor_keys`.
- Risolvere ogni `ActiveOrder` con il resolver centrale.
- Emissione minima: `actor_key`, `actor_id`, `identity_level`, `identity_source`, `identity_fallback_flag`.
- Rinominare le colonne metriche `client_*` in `actor_*` negli artefatti nuovi; se serve una transizione, mantenere alias legacy soltanto in una funzione adapter esplicitamente marcata e non negli output ufficiali.
- Aggiornare `_same_level_visible_qty` affinché filtri con `actor_key`, non con `order.client_original_id`.

**Step 4: Verificare regressione numerica client-level**

Usare gli stessi dati sintetici dei test correnti e confrontare DWI, liquidità, SCI e zero-denominator semantics con i valori precedenti, non soltanto shape/assenza di crash.

Run:

```bash
pytest tests/lob/test_spoofing_metrics.py -q -k "top_n_exposures or zero"
```

Expected: PASS; valori client-level invariati entro le tolleranze pytest già usate.

---

### Task 3: Generalizzare il clusterer a identità attore e due execution anchor

**Objective:** Costruire cluster deterministici per fill passive e aggressive, mantenendo provenance raw e separazione di ramo.

**Files:**
- Modify: `src/spoofing_detection/lob/execution_clusters.py:1-260`
- Modify: `tests/lob/test_execution_clusters.py`

**Step 1: Scrivere test RED di classificazione ruolo**

Testare:

```python
assert classify_execution_anchor({"PASSIVEORDER": "Y", "AGGRESSIVEORDER": "N"}) == "passive"
assert classify_execution_anchor({"PASSIVEORDER": None, "AGGRESSIVEORDER": "Y"}) == "aggressive"
assert classify_execution_anchor({"PASSIVEORDER": "Y", "AGGRESSIVEORDER": "Y"}) is None
```

La funzione di candidatura, non il solo classifier, deve associare il motivo `ambiguous_execution_role` al terzo caso.

**Step 2: Scrivere test RED per clustering aggressive multi-price**

Costruire due child fill dello stesso ordine aggressivo, stesso actor/partition/side, entro 100 ms, con `LASTTRADEDPX` diversi. Verificare:

```python
assert len(clusters) == 1
assert clusters[0]["execution_anchor_mode"] == "aggressive"
assert clusters[0]["child_fill_count"] == 2
assert clusters[0]["event_price"] == pytest.approx(expected_vwap)
assert clusters[0]["execution_cluster_id"].startswith("EC-A-")
```

Aggiungere test che un cambio di attore, ordine, partition, side, ramo o gap chiuda il cluster. Per passive, conservare la separazione per prezzo già esistente; per aggressive, consentire lo sweep multi-price.

**Step 3: Verificare RED**

Run:

```bash
pytest tests/lob/test_execution_clusters.py -q
```

Expected: FAIL sui nuovi casi aggressive/actor.

**Step 4: Implementare un clusterer generico**

- Sostituire `PendingExecutionCluster.client_original_id` con campi actor e `execution_anchor_mode`.
- Conservare `weighted_price_notional` per VWAP.
- Separare:
  - `_base_fill_identity`: partition, order, actor, side, anchor;
  - regola price-sensitive per passive;
  - regola price-insensitive per aggressive.
- Esporre `cluster_execution_fills(events, allowed_anchor_modes, max_gap_ms)`.
- Lasciare `cluster_passive_execution_fills` solo come wrapper di compatibilità se un consumer non migrato lo richiede; gli script ufficiali devono usare la nuova API.
- Aggiungere `child_execution_anchor_mode`, actor fields e identificativi grezzi ai member rows.

**Step 5: Verificare PASS e determinismo**

Run:

```bash
pytest tests/lob/test_execution_clusters.py -q
```

Expected: PASS; ordinamento cluster/member stabile a input identico.

---

### Task 4: Accettare i fill firm fallback e aggressive nella scansione LOB

**Objective:** Rimuovere le due esclusioni strutturali senza indebolire validazioni di timestamp, quantità, prezzo o side.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:364-524`
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:676-805`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Scrivere il caso sintetico Ferrari-like RED**

Creare una sequenza con stesso `FIRMID`, client mancante:

1. ordine candidato recente sul deceptive side;
2. piccolo ordine passivo sul lato opposto;
3. fill passivo;
4. cancellazione rapida dell’ordine candidato.

Verificare:

```python
assert result.rejected_executions.filter(pl.col("reject_reason") == "missing_client_id").is_empty()
row = result.execution_metrics.row(0, named=True)
assert row["actor_key"] == "firm:157922_3"
assert row["identity_level"] == "firm"
assert row["identity_fallback_flag"] is True
assert row["has_matched_deceptive_cancel_window"] is True
```

Aggiungere un caso negativo in cui fill e ordine candidato hanno firm diverse.

**Step 2: Scrivere il caso sintetico Nexi-like RED**

Sequenza minima:

1. ordine/depth dello stesso client sul deceptive side;
2. movimento pre-fill coerente;
3. fill aggressive con `LASTSHARES`, `LASTTRADEDPX`, anche order type market;
4. cancellazione rapida del profilo candidato;
5. stato post-cancel per reversion.

Verificare che il cluster sia presente e branch-labeled; non imporre nel primo test che tutti i gate siano veri. Aggiungere un secondo test analitico con prezzi costruiti per far superare consapevolmente i quattro gate.

**Step 3: Scrivere casi RED di rifiuto**

- entrambi gli identificativi mancanti → `missing_actor_identity`;
- aggressive senza `LASTTRADEDPX` → `missing_execution_price`;
- aggressive senza `LASTSHARES` positivo → `missing_or_invalid_fill_qty`;
- entrambi i role flag `Y` → `ambiguous_execution_role`;
- passive senza ordine attivo → rifiuto corrente conservato;
- aggressive senza ordine attivo → accettato.

**Step 4: Implementare la selezione branch-aware**

Refactor proposto:

```python
def _execution_candidate_or_rejection(
    event,
    active_orders,
    *,
    event_ts,
    partition_id,
    allowed_anchor_modes,
):
    actor = actor_identity_from_event(event)
    anchor = classify_execution_anchor(event)
    ...
```

- Nel ramo passive mantenere i controlli di active order e visible limit.
- Nel ramo aggressive non richiedere active order e non rifiutare market/crossing limit.
- Per aggressive usare esclusivamente `LASTTRADEDPX` e `LASTSHARES` come prezzo/quantità child-fill.
- Inserire `execution_price_source` e actor fields nella base row e nel rejected row.
- Non mutare la logica LOB di `panel.py`: la ricostruzione gestisce già ordini aggressivi/residui e va verificata, non riscritta.
- Catturare il profilo candidato dallo stato pre-evento, prima di `_apply_event`.
- Lasciare `smallness_fraction_*` passive-specific; per aggressive emettere `null` con metadata che ne chiarisca la non applicabilità.

**Step 5: Verificare i test focalizzati**

Run:

```bash
pytest tests/lob/test_spoofing_metrics.py -q -k "firm_fallback or aggressive or execution_candidate"
```

Expected: PASS.

---

### Task 5: Migrare candidature, cancellazioni, finestre e gate a `actor_key`

**Objective:** Garantire che la nuova identità sia usata end-to-end e non soltanto sulla riga di execution.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:449-930`
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:989-1505`
- Modify: `src/spoofing_detection/lob/behavioral_gate.py`
- Modify: `tests/lob/test_spoofing_metrics.py`
- Modify: `tests/lob/test_behavioral_gate.py`

**Step 1: Scrivere test RED anti-cross-attribution**

Coprire separatamente:

- stesso raw id ma namespace diverso → nessun match;
- stessa firm fallback → match;
- client execution con ordine same-firm ma client mancante → nessun match cross-level;
- una cancellazione fisica candidabile per cluster passive e aggressive viene assegnata una sola volta dentro lo stesso ramo/episodio secondo la regola canonica; se la competizione cross-branch è possibile, la chiave di assegnazione deve includere `actor_key` e la regola deve essere documentata.

**Step 2: Sostituire i confronti client-only**

Aggiornare:

- `_candidate_deceptive_order_rows`;
- `_direct_cancel_row`;
- lookup/grouping dello state time series;
- `_build_execution_cancel_candidates`;
- candidate schema `EXECUTION_CANCEL_CANDIDATE_SCHEMA`;
- cancellazioni dirette e candidate rows;
- `attach_sci_window_metrics`;
- output `spoofing_compatible_events`.

Ogni join deve usare `actor_key`; `identity_level` e `execution_anchor_mode` devono essere propagati e verificati, non ricostruiti da stringhe a valle.

**Step 3: Rendere branch-aware l’assegnazione delle cancellazioni**

Decisione operativa iniziale:

- una cancellazione fisica non deve essere contata due volte per lo stesso attore e ordine;
- tra cluster eligible, mantenere la regola `latest_prior_cluster_end` indipendentemente dal ramo;
- includere nei candidati il ramo vincente e il numero di cluster concorrenti passive/aggressive;
- aggiungere test che provi esattamente questa proprietà.

**Step 4: Verificare gate e invarianti**

Il gate resta:

```text
rapid attributed cancellation
AND fill_qty < withdrawn_qty
AND favorable pre-fill mid move
AND positive cancel-anchored mid reversion
```

Aggiungere test parametrico `execution_anchor_mode in {passive, aggressive}` con gli stessi valori metrici, verificando uguale semantica booleana e conservazione del ramo.

Run:

```bash
pytest tests/lob/test_spoofing_metrics.py tests/lob/test_behavioral_gate.py -q
```

Expected: PASS.

---

### Task 6: Separare scoring e production readiness per attore e ramo

**Objective:** Evitare che eventi aggressive alterino impropriamente denominatori, ranking e soglie del ramo passive.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:1507-1650`
- Modify: `src/spoofing_detection/lob/client_session_features.py`
- Modify: `src/spoofing_detection/lob/legitimacy_features.py`
- Modify: `src/spoofing_detection/lob/alert_objects.py`
- Modify: `tests/lob/test_client_session_features.py`
- Modify: `tests/lob/test_legitimacy_features.py`
- Modify: `tests/lob/test_alert_objects.py`

**Step 1: Scrivere test RED per la separazione dei denominatori**

Input sintetico dello stesso attore con due cluster passive e tre aggressive. Verificare due righe aggregate:

```python
assert set(features["execution_anchor_mode"]) == {"passive", "aggressive"}
assert passive_row["event_count"] == 2
assert aggressive_row["event_count"] == 3
```

Aggiungere un attore firm-level e verificare che `identity_level=firm` arrivi fino all’alert.

**Step 2: Migrare MCPS**

- Grouping: `(actor_key, execution_anchor_mode, top_n, gamma)`.
- Conservare `actor_id`, `identity_level`, source e fallback flag tramite aggregazioni con verifica di unicità.
- Rinominare `all_attributable_client_execution_clusters` in metadata con formulazione actor/branch corretta.
- Vietare input duplicati su `(execution_cluster_id, execution_anchor_mode)`.

**Step 3: Migrare feature e alert**

Esporre API canoniche:

```python
compute_actor_session_features(...)
compute_actor_legitimacy_features(...)
build_actor_session_alerts(...)
```

Aggiornare gli schema a `actor_key` + branch. Eventuali wrapper `compute_client_*` possono esistere soltanto per fixture/consumer legacy e devono rifiutare firm fallback invece di rinominarlo client.

**Step 4: Trattare la qualità dell’identità esplicitamente**

Negli alert aggiungere almeno:

- `identity_level`;
- `identity_fallback_flag`;
- `identity_scope_warning`, valorizzato per firm fallback;
- `execution_anchor_mode`.

Non introdurre un punteggio arbitrario di confidence. La raccomandazione resta human review; il warning esplicita che il gruppo firm-level può contenere più clienti.

**Step 5: Verificare**

Run:

```bash
pytest tests/lob/test_client_session_features.py tests/lob/test_legitimacy_features.py tests/lob/test_alert_objects.py -q
```

Expected: PASS e nessun pooling dei due rami.

---

### Task 7: Esporre configurazione, schema version e nuovi artefatti CLI

**Objective:** Rendere il cambiamento riproducibile, configurabile e impossibile da confondere con run client/passive legacy.

**Files:**
- Modify: `configs/spoofing_detection_parameters.json`
- Modify: `src/spoofing_detection/lob/spoofing_config.py:10-20`
- Modify: `scripts/compute_spoofing_metrics.py`
- Modify: `scripts/run_multilevel_spoofing_grid.py`
- Modify: `scripts/compute_client_session_spoofing_features.py`
- Modify: `scripts/run_spoofing_production_readiness.py`
- Modify: `tests/lob/test_compute_spoofing_metrics_cli.py`
- Modify or create: `tests/lob/test_run_multilevel_spoofing_grid.py`
- Modify: CLI test di production readiness se già presente; altrimenti create `tests/lob/test_run_spoofing_production_readiness.py`

**Step 1: Scrivere test RED config/CLI**

Nuovi parametri:

```json
{
  "actor_identity_mode": "client_then_firm",
  "execution_anchor_modes": ["passive", "aggressive"]
}
```

CLI:

```text
--actor-identity-mode client_then_firm
--execution-anchor-modes passive,aggressive
```

Testare config default, override CLI, valore vuoto, ramo sconosciuto, duplicati e ordine canonico.

**Step 2: Implementare parsing senza nuove dipendenze**

- Estendere la normalizzazione list-to-CSV della config anche a `execution_anchor_modes`.
- Validare il set contro `{"passive", "aggressive"}`.
- Per invocazioni senza config, mantenere un default CLI esplicito e documentato; la config ufficiale può abilitare entrambi i rami.
- Passare i valori fino a `compute_exploratory_metrics`.

**Step 3: Versionare metadata e cache reuse**

Aggiungere:

```json
{
  "output_schema_version": "actor_execution_anchor_v1",
  "actor_identity_mode": "client_then_firm",
  "execution_anchor_modes": ["passive", "aggressive"],
  "score_grouping": ["actor_key", "execution_anchor_mode"],
  "firm_fallback_semantics": "aggregate only when client_original_id is missing"
}
```

Includere questi campi in `expected_metadata` di `run_multilevel_spoofing_grid.py`, così `--reuse-depth-outputs` non può riusare run passive/client legacy.

**Step 4: Rinominare gli artefatti ufficiali**

Scrivere nuovi nomi, senza sovrascrivere run precedenti:

- `actor_metric_time_series.parquet`;
- `actor_mcps_scores.parquet`;
- `combined_actor_mcps_scores.parquet`;
- `actor_session_features.parquet`;
- `actor_session_risk_features.parquet`;
- `actor_legitimacy_features.parquet`;
- `actor_session_alerts.parquet`.

I nomi event-level neutrali (`execution_metrics.parquet`, `candidate_deceptive_orders.parquet`, ecc.) possono restare, ma devono avere schema version nei metadata.

**Step 5: Aggiornare summary e audit counts**

Riportare separatamente:

- cluster passive/aggressive;
- client-level/firm-fallback;
- rejected by reason;
- matched withdrawal e strict sequence per ramo/livello;
- quantità di righe senza entrambe le identità.

Non ordinare insieme score firm e client come se fossero direttamente comparabili; tabelle top devono essere stratificate almeno per `identity_level` e `execution_anchor_mode`.

**Step 6: Verificare CLI**

Run:

```bash
pytest tests/lob/test_compute_spoofing_metrics_cli.py tests/lob/test_run_multilevel_spoofing_grid.py tests/lob/test_run_spoofing_production_readiness.py -q
```

Expected: PASS.

---

### Task 8: Aggiornare dashboard, dossier e audit operativi

**Objective:** Rendere visibili identità e ramo senza rompere le sezioni investigative essenziali.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metric_plots.py`
- Modify: `scripts/plot_spoofing_metrics.py`
- Modify: `scripts/build_spoofing_event_dossier.py`
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `scripts/audit_execution_cluster_outputs.py`
- Modify: relativi test sotto `tests/lob/`, in particolare `tests/lob/test_spoofing_event_review_dashboard.py`

**Step 1: Scrivere test RED delle colonne/label**

Verificare che filtri e testate mostrino:

- `Actor` come `actor_id`;
- `Identity level`;
- warning firm fallback;
- `Execution anchor`;
- client/firm grezzi nei pannelli provenance;
- nessuna label “client” applicata a `identity_level=firm`.

**Step 2: Migrare filtri e join**

Sostituire i filtri `client_id` con `actor_key`. Conservare le sezioni già essenziali:

- event selection;
- LOB pre/post;
- event-window tables;
- candidate/cancel provenance;
- annotazioni e review;
- righe evidenziate.

Non duplicare metriche già mostrate: aggiungere identity/branch nella testata e nella tabella evento, non in ogni pannello.

**Step 3: Aggiornare dossier provenance**

In `build_spoofing_event_dossier.py`, estendere la lista identificativa che oggi include `event_client_original_id`/`client_id` con actor fields, raw firm e `execution_anchor_mode`.

**Step 4: Verificare**

Run:

```bash
pytest tests/lob/test_spoofing_event_review_dashboard.py tests/lob/test_spoofing_metric_plots.py -q
```

Se il secondo file non esiste, aggiungere test focalizzati nel file di test plot già presente, senza creare una suite duplicata.

---

### Task 9: Documentare metodo e limiti senza ancora riscrivere il paper

**Objective:** Allineare documentazione operativa e impedire claim non validati sul nuovo ramo.

**Files:**
- Modify: `docs/spoofing_production_readiness.md`
- Modify: `docs/keep_cols_data_dictionary.md:95-105`
- Modify: `docs/lob_implementation_status.md`
- Do not modify yet: `paper/spoofing.tex`
- Do not regenerate yet: `paper/generated/spoofing_empirical_tables.tex`
- Do not modify yet: `scripts/generate_spoofing_paper_tables.py`

**Step 1: Documentare identity fallback**

Spiegare:

- gerarchia client-then-firm;
- namespace;
- assenza di cross-level attribution;
- firm fallback come aggregato meno granulare;
- stratificazione degli output.

**Step 2: Documentare aggressive role**

Inserire la definizione operativa di role flag e il requisito `LASTTRADEDPX`/`LASTSHARES`. Richiamare esplicitamente l’incertezza ETL già descritta nel data dictionary e il controllo che deve precedere conclusioni empiriche.

**Step 3: Mantenere il manoscritto baseline invariato**

Il manoscritto contiene numerose occorrenze che definiscono il modello come passive/client-only (`paper/spoofing.tex`, fra cui righe 60, 93-95, 292-300, 852-877, 968-1026, 1084-1129, 1146-1171). Non aggiornarle prima del rerun e della review scientifica: il ramo aggressive deve essere inizialmente riportato come estensione sperimentale separata.

Dopo validazione, creare un piano paper dedicato che:

- separi modifiche manoscritto da generator-level;
- aggiorni tutte le occorrenze visibili;
- modifichi `scripts/generate_spoofing_paper_tables.py` prima di rigenerare `paper/generated/spoofing_empirical_tables.tex`;
- rigeneri tabelle/figure;
- compili e ispezioni il PDF.

---

### Task 10: Eseguire test di regressione e due review avversariali

**Objective:** Verificare software, invarianti scientifiche e assenza di regressioni nel ramo passivo.

**Files:**
- Modify only if findings require fixes: files touched in Tasks 1-9
- No generated empirical output committed

**Step 1: Suite focalizzata core**

Run:

```bash
pytest \
  tests/lob/test_actor_identity.py \
  tests/lob/test_execution_clusters.py \
  tests/lob/test_spoofing_metrics.py \
  tests/lob/test_behavioral_gate.py \
  -q
```

Expected: PASS.

**Step 2: Suite production/UI**

Run:

```bash
pytest \
  tests/lob/test_client_session_features.py \
  tests/lob/test_legitimacy_features.py \
  tests/lob/test_alert_objects.py \
  tests/lob/test_compute_spoofing_metrics_cli.py \
  tests/lob/test_spoofing_event_review_dashboard.py \
  -q
```

Expected: PASS.

**Step 3: Suite completa**

Run:

```bash
pytest -q
```

Expected: PASS. Se l’ambiente attivo non contiene Polars/PyArrow, usare l’ambiente di progetto già previsto; non installare globalmente e non cambiare dependency manager senza manifest.

**Step 4: Review scientific/specification**

Checklist:

- nessun actor cross-level;
- nessun pooling passive/aggressive;
- nessun firm fallback etichettato client;
- aggressive price/qty solo da campi trade effettivi;
- nessun double count di cancellazioni;
- cluster sweep deterministico;
- gate e segni price-move/reversion coerenti con bid/ask;
- zero-denominator semantics invariate;
- risultati descritti come surveillance cues, non intent labels.

**Step 5: Review qualità/regressione**

Cercare:

```bash
# usare search_files, non grep, durante l'esecuzione
```

Pattern da eliminare dai percorsi canonici: grouping/join su `client_id`, metadata `all_passive_execution_clusters`, label “Top clients” non stratificata, `aggressive_execution` come rejection per righe valide quando il ramo è abilitato.

Controllare `git diff --check` e diff limitato ai file pianificati. Non fare commit senza autorizzazione esplicita dell’utente.

---

### Task 11: Rerun empirico isolato e verifica degli alert esterni

**Objective:** Dimostrare che i due buchi strutturali sono chiusi sui dati reali senza sovrascrivere gli output correnti o forzare il risultato finale.

**Files:**
- Create runtime only: nuova directory timestamped sotto `outputs/spoofing_metrics/`
- Optional create: `scripts/audit_external_alert_coverage.py`
- Optional create: `tests/lob/test_audit_external_alert_coverage.py`
- Input sensibile esterno: non committare PDF/CSV con alert o identificativi se non autorizzato

**Step 1: Smoke run sintetico CLI**

Eseguire `scripts/compute_spoofing_metrics.py` su un piccolo parquet sintetico contenente un caso passive/client, passive/firm e aggressive/client. Verificare schema, metadata e contatori.

Expected:

- tre cluster distribuiti correttamente per identity/branch;
- zero collisioni;
- rejected reasons soltanto per righe intenzionalmente invalide.

**Step 2: Audit semantico dei role flag reali**

Prima del full rerun, per ogni titolo riportare la tabella incrociata:

```text
PASSIVEORDER × AGGRESSIVEORDER × event_order_type_label × presenza LASTTRADEDPX/LASTSHARES
```

Verificare anche il pairing per `EXECUTIONID`/`TRADEUNIQUEIDENTIFIER`. Se i quattro fill Nexi risultano ambigui o senza trade price/qty, fermare il ramo aggressive e documentare il blocker invece di usare `ORDERPX`/`ORDERQTY` come sostituti inventati.

**Step 3: Full rerun in directory nuova**

Usare gli stessi parametri della run baseline (`top10`, `h10`, age 90, empirical kernel, exact piecewise ratios) e cambiare soltanto:

- schema actor fallback;
- execution anchor modes passive+aggressive.

Registrare command, config, hashes input/kernel, package versions e output paths nei metadata. Non sovrascrivere:

`outputs/spoofing_metrics/20260723_132330_empirical_kernel_top10_h10_age90_exact_piecewise_ratios/`

**Step 4: Reconciliation alert-level**

Se viene aggiunto `audit_external_alert_coverage.py`, l’interfaccia deve accettare un CSV non tracciato con:

```text
instrument, external_actor_id, start_ts, end_ts, expected_identity_level
```

Output per finestra:

- execution child rows osservate;
- cluster eligible per branch/identity;
- rejected reasons;
- candidate profile count;
- matched cancellation;
- strict gate;
- classificazione `eligible`, `matched`, `strict`, `rejected`, `not_observable`.

Non codificare nel repository gli identificativi del documento interno.

**Step 5: Acceptance empirica minima**

- Ferrari: le esecuzioni nelle 35 finestre devono essere riconducibili a `actor_key=firm:<FIRMID>` quando il firm è presente; il conteggio `missing_actor_identity` deve essere zero in quelle finestre.
- Nexi: i quattro fill aggressivi devono apparire nel cluster/member output con ramo aggressive, non nel bucket `aggressive_execution`.
- Risanamento: non è criterio per questi due interventi senza timestamp esterno preciso, ma il client-level passive baseline deve restare disponibile.
- Separare chiaramente “ammesso dal generatore” da “matched cancel” e “strict sequence”. Un caso può essere correttamente recuperato dall’intervento e continuare a non superare i gate.

**Step 6: Confronto regressione baseline**

Sul sottoinsieme `identity_level=client_original AND execution_anchor_mode=passive`, confrontare nuova e vecchia run per:

- numero child fill e cluster;
- membership cluster;
- candidate order ids;
- cancellation assignment;
- MSCI/WMSCI e gate;
- MCPS per client/top-N/gamma.

Expected: uguaglianza esatta per ID/count dove matematicamente prevista e `pytest.approx`/tolleranza esplicita soltanto per float. Ogni differenza va spiegata prima di accettare il rerun.

---

## 5. Files likely to change

### Core

- `src/spoofing_detection/lob/actor_identity.py` — nuovo resolver canonico.
- `src/spoofing_detection/lob/models.py` — eventuali helper tipizzati.
- `src/spoofing_detection/lob/execution_clusters.py` — clustering actor/branch-aware.
- `src/spoofing_detection/lob/spoofing_metrics.py` — candidatura, stato, cancellazioni, score.
- `src/spoofing_detection/lob/behavioral_gate.py` — conservazione ramo/identity.
- `src/spoofing_detection/lob/client_session_features.py` — migrazione ad actor session.
- `src/spoofing_detection/lob/legitimacy_features.py` — aggregazione actor-aware.
- `src/spoofing_detection/lob/alert_objects.py` — alert actor+branch.
- `src/spoofing_detection/lob/spoofing_metric_plots.py` — label/filtri.

### CLI/config/operational consumers

- `configs/spoofing_detection_parameters.json`
- `src/spoofing_detection/lob/spoofing_config.py`
- `scripts/compute_spoofing_metrics.py`
- `scripts/run_multilevel_spoofing_grid.py`
- `scripts/compute_client_session_spoofing_features.py`
- `scripts/run_spoofing_production_readiness.py`
- `scripts/plot_spoofing_metrics.py`
- `scripts/build_spoofing_event_dossier.py`
- `scripts/build_spoofing_event_review_dashboard.py`
- `scripts/audit_execution_cluster_outputs.py`
- opzionale `scripts/audit_external_alert_coverage.py`

### Tests

- nuovo `tests/lob/test_actor_identity.py`
- `tests/lob/test_execution_clusters.py`
- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_behavioral_gate.py`
- `tests/lob/test_client_session_features.py`
- `tests/lob/test_legitimacy_features.py`
- `tests/lob/test_alert_objects.py`
- `tests/lob/test_compute_spoofing_metrics_cli.py`
- test grid/production/dashboard corrispondenti

### Documentation

- `docs/spoofing_production_readiness.md`
- `docs/keep_cols_data_dictionary.md`
- `docs/lob_implementation_status.md`

### Esplicitamente fuori dal primo intervento

- `paper/spoofing.tex`
- `paper/generated/spoofing_empirical_tables.tex`
- `scripts/generate_spoofing_paper_tables.py`

Questi ultimi richiedono un piano manuscript/generator dedicato dopo validazione empirica.

## 6. Rischi e trade-off

1. **Pooling firm-level:** il fallback può unire più client reali; mitigazione: namespace, warning, output stratificati e nessun claim client-level.
2. **Comparabilità score:** MCPS passive e aggressive non sono immediatamente comparabili; mitigazione: grouping/alert separati e nessuna soglia pooled.
3. **Semantica ETL role flags:** `PASSIVEORDER`/`AGGRESSIVEORDER` potrebbero derivare da qualifier diversi; mitigazione: audit reale prima del rerun.
4. **Double counting delle due gambe del trade:** i feed possono contenere una riga passive e una aggressive per la stessa esecuzione; mitigazione: provenance per `EXECUTIONID`/trade UID e cluster per actor/branch, con test paired-row.
5. **Sweep multi-price:** una marketable order può eseguire su più livelli; mitigazione: cluster per ordine/actor/side/gap e VWAP, non per prezzo.
6. **Metriche passive-specific:** same-level client smallness non è trasferibile automaticamente; mitigazione: `null` branch-aware, senza epsilon o proxy arbitrari.
7. **Collisioni ID grezzi:** stesso valore in namespace client/firm; mitigazione: `actor_key` namespaced.
8. **Cache/output stale:** vecchi output hanno schema client/passive; mitigazione: schema version nei metadata e nomi actor-specific.
9. **Regressione del paper:** il manoscritto definisce ancora un modello passive/client; mitigazione: mantenere l’estensione sperimentale separata finché il rerun non è validato.
10. **Scope:** non aggiungere nuovi modelli, threshold tuning o intent labels in questo intervento; YAGNI.

## 7. Open questions da chiudere durante Task 11, non da indovinare

- I quattro fill Nexi hanno sempre `LASTTRADEDPX` e `LASTSHARES` affidabili?
- I role flag sono mutuamente esclusivi nelle righe reali o esiste una quota significativa `Y/Y`?
- La stessa cancellazione può competere concretamente fra cluster passive e aggressive dello stesso attore? Se sì, la regola globale `latest_prior_cluster_end` resta la scelta conservativa iniziale.
- Quanti clienti distinti risultano talvolta visibili sotto lo stesso `FIRMID`? Serve per quantificare il rischio di aggregazione firm-level, non per cambiare il fallback.
- Dopo il recupero della popolazione, quali alert esterni superano soltanto eligibility, quali matched withdrawal e quali strict gate? Nessuna di queste classi va presupposta.

## 8. Definition of done

- Tutti i test focalizzati e `pytest -q` passano.
- Il ramo passivo client-level è numericamente coerente con la baseline.
- Il fallback firm-level è visibile, namespaced e mai presentato come client.
- Le execution aggressive valide producono cluster auditabili con trade price/qty effettivi.
- Score e alert sono separati per `execution_anchor_mode`.
- Metadata e artefatti rendono il nuovo schema inequivocabile e impediscono reuse stale.
- Il rerun reale chiude le due cause strutturali osservate (`missing_client_id`, `aggressive_execution`) senza alterare le soglie per ottenere un risultato desiderato.
- Il report finale distingue eligibility, matched cancellation, strict sequence, falsi negativi residui e incertezza; non usa gli alert esterni come prova di manipolazione.
