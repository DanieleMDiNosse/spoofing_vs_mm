# Execution clustering, review LLM e paper scientifico — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** sostituire i fill message-level duplicati con execution cluster auditabili, associare ogni cancellazione a un solo cluster, rigenerare dossier/review RISANAMENTO con un contratto evidenziario prudente e aggiornare il paper con definizioni e risultati riproducibili.

**Architecture:** mantenere intatti i messaggi raw e introdurre tre livelli espliciti: `fill message` (provenance), `execution cluster` (unità primaria delle metriche), `client-session` (ripetizione). I cluster vengono costruiti prima del matching delle cancellazioni; l’associazione canonica di ogni cancellazione usa una regola uno-a-uno documentata. Dossier, dashboard, batch LLM e manoscritto consumano poi gli stessi artefatti cluster-level, mentre tabelle e macro numeriche del paper sono generate dagli output validati invece di essere ricopiate a mano.

**Tech Stack:** Python, Polars, pytest, JSON/Parquet/CSV, HTML dashboard, API LLM già configurata, LaTeX/TikZ/BibTeX.

---

## 1. Contesto osservato e assunzioni

- Repository: `/home/danielemdn/Documents/repositories/spoofing_detection`.
- Il working tree contiene numerose modifiche preesistenti. Durante l’esecuzione del piano occorre rileggere ogni file prima di modificarlo, non sovrascrivere cambiamenti non correlati e non fare commit senza autorizzazione esplicita dell’utente.
- La pipeline corrente crea una riga in `execution_metrics` per ogni fill passivo eleggibile in `src/spoofing_detection/lob/spoofing_metrics.py`; `matched_spoofing_events` e i conteggi client-session ereditano quindi la granularità message-level.
- Il caso di regressione `S101248` contiene cinque fill passivi dello stesso ordine `2214678037`, agli indici raw `101248`, `101251`, `101254`, `101257`, `101260`, con quantità `9000`, `9000`, `1947`, `857`, `4196`. I messaggi sono intervallati dalle gambe aggressive e da nuovi ordini aggressivi, quindi non sono adiacenti nel flusso raw; la quantità cluster attesa è `25000`.
- Il rapporto atteso per una cancellazione da `219770` contro il cluster da `25000` è `8.7908`, non cinque rapporti costruiti su denominatori child-fill.
- `src/spoofing_detection/lob/panel.py::_fill_group_key` identifica le due gambe dello stesso trade tramite `EXECUTIONID`, `TRADEUNIQUEIDENTIFIER`, `TRADETIME` e prezzo. Non va riutilizzato come identità del nuovo cluster: i cinque child fill hanno execution ID diversi.
- Gli output canonici correnti usano `outputs/spoofing_metrics/20260714_rank_pooled_empirical_kernel_top5_h10_age90/`; la nuova pipeline deve scrivere in una directory run nuova e non sovrascriverli.
- Il paper contiene tabelle empiriche hard-coded e figure TikZ inline. Non risultano `\includegraphics` o `\input` attivi: per la sincronizzazione numerica va introdotto un piccolo generatore di macro/tabelle, mentre le modifiche visive alle figure TikZ restano localizzate in `paper/spoofing.tex`.
- L’interprete Python di sistema non ha Polars. Inoltre `conda run -n main` ha mostrato uno stato di attivazione incoerente; i comandi riproducibili del piano usano direttamente `/home/danielemdn/miniconda3/envs/main/bin/python` finché l’ambiente non viene sistemato separatamente.
- I punteggi, i match meccanici e le review LLM restano strumenti di sorveglianza esplorativa: nessuno di essi prova intento manipolativo.

## 2. Semantica scientifica da fissare prima dei risultati

### 2.1 Tre unità distinte

1. **Fill message:** singolo record normalizzato; preservato integralmente per audit.
2. **Execution cluster:** sequenza di fill passivi dello stesso ciclo di ordine, stesso client, lato e prezzo, separati al massimo da un parametro operativo `execution_cluster_max_gap_ms`. Pre-book e candidate posture sono ancorati al primo child fill; la quantità eseguita è la somma dei child fill; la finestra di cancellazione parte dall’ultimo child fill.
3. **Client-session:** aggregazione di cluster unici; denominatori MCPS/MES e conteggi di ripetizione non usano i child fill.

### 2.2 Regola operativa iniziale e sensitivity gate

- Aggiungere `execution_cluster_max_gap_ms` alla configurazione, con valore operativo iniziale `100` ms.
- Unire due fill passivi consecutivi dello stesso ordine solo se:
  - appartengono alla stessa partizione LOB;
  - hanno stesso `ORDERID`, client originale, lato e prezzo;
  - non esiste tra loro un evento lifecycle sullo stesso ordine che lo modifichi, cancelli o ricarichi;
  - il gap fra i timestamp dei fill passivi è `<= execution_cluster_max_gap_ms`.
- Gli eventi della gamba aggressiva e i nuovi ordini aggressivi tra due child fill non spezzano il cluster.
- Chiudere il cluster su fill terminale (`LEAVESQTY <= 0`), cambio partizione, evento lifecycle incompatibile, superamento del gap o fine stream.
- Il valore `100` ms è un parametro operativo, non una costante economica universale. Prima di usare i risultati nel paper, confrontare almeno `25`, `50`, `100`, `250`, `1000` ms. Se conteggi, ranking o conclusioni non sono stabili, riportare la sensibilità e ridurre la forza delle conclusioni invece di scegliere ex post la soglia più favorevole.

### 2.3 Associazione cancellazione → cluster

- Conservare tutte le associazioni eleggibili in un artefatto audit (`execution_cancel_candidates`).
- Per le metriche canoniche, assegnare ogni cancellazione identificata da `(partition_id, cancel_sort_index, ORDERID)` a un solo cluster: quello con `cluster_end_ts` più recente fra i cluster eleggibili precedenti alla cancellazione entro la finestra; usare `cluster_first_sort_index` come tie-breaker deterministico.
- WMSCI, withdrawal-to-fill ratio, `matched_event_count`, MES, alert score e dashboard devono usare solo associazioni assegnate.
- Preservare nel file di link `assignment_rule = latest_prior_cluster_end` e il numero di cluster concorrenti, così l’ambiguità resta ispezionabile.

### 2.4 Temporizzazione cluster-level

- `pre` LOB e candidate posture: istante immediatamente precedente al primo child fill.
- Prezzo cluster: media ponderata per quantità dei child fill; se i prezzi differiscono nonostante l’assenza di modify, emettere un flag di qualità e non nascondere la discrepanza.
- `cluster_end_ts`: timestamp dell’ultimo child fill.
- Delay di cancellazione e finestra post-evento: misurati da `cluster_end_ts`.
- FPM: dal primo posting della candidate posture al pre-state del cluster.
- REV e post-state: dopo la cancellazione assegnata/finestra successiva al cluster, secondo l’attuale contratto esplicitato nei metadati.

## 3. Output contract previsto

### 3.1 Nuovi artefatti

- `execution_cluster_members.parquet` / `.csv`: una riga per child fill con `execution_cluster_id`, `child_event_id`, `child_sort_index`, `child_execution_id`, `child_trade_uid`, quantità, prezzo e timestamp.
- `execution_cancel_candidates.parquet` / `.csv`: tutti i link cluster-cancellazione eleggibili, con `assigned_flag`, regola e numero di concorrenti.
- `execution_metrics.parquet` / `.csv`: una riga per cluster, non per fill message.

### 3.2 Campi cluster minimi

```python
CLUSTER_COLUMNS = {
    "execution_cluster_id",
    "cluster_first_event_id",
    "cluster_last_event_id",
    "cluster_first_sort_index",
    "cluster_last_sort_index",
    "cluster_start_ts",
    "cluster_end_ts",
    "child_fill_count",
    "event_order_id",
    "event_client_original_id",
    "event_side",
    "event_price",
    "fill_qty",
    "execution_cluster_gap_ms",
    "execution_cluster_quality_flags",
}
```

- ID deterministico suggerito: `EC{cluster_first_sort_index:09d}-{cluster_last_sort_index:09d}`. Il `partition_id` resta una colonna obbligatoria; se gli indici non sono globalmente unici, includere un prefisso partizione normalizzato e testarlo.
- `review_event_id` deve essere il cluster ID. I vecchi ID `S...` restano solo in `execution_cluster_members` e non devono essere usati per incorporare automaticamente vecchie review nei nuovi output.

## 4. Piano step-by-step

### Task 1: Proteggere baseline e scrivere i test di regressione del caso reale

**Objective:** fissare invarianti e failure mode prima di implementare il clustering.

**Files:**
- Modify: `tests/lob/test_spoofing_metrics.py`
- Test fixture source: `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet` (read-only)

**Step 1: rileggere stato e diff prima di ogni edit**

Run:
```bash
git status --short
git diff -- src/spoofing_detection/lob/spoofing_metrics.py tests/lob/test_spoofing_metrics.py
```
Expected: stato corrente osservabile; nessuna modifica automatica o stage di file estranei.

**Step 2: aggiungere una fixture sintetica minima**

La fixture deve rappresentare:
- un ordine passivo da `25000`;
- cinque child fill `9000 + 9000 + 1947 + 857 + 4196`;
- gambe aggressive e nuovi ordini aggressivi interposti;
- una candidate ask già parzialmente eseguita e poi cancellata;
- un evento di controllo oltre il gap che deve formare un altro cluster.

**Step 3: scrivere test failing**

Aggiungere test separati per:
- `child_fill_count == 5`;
- `fill_qty == 25000`;
- pre-state ancorato al primo fill;
- end timestamp ancorato all’ultimo fill;
- un solo match alla cancellazione;
- ratio `pytest.approx(8.7908)` per quantità cancellata `219770`;
- split se gap `> 100 ms`;
- split su modify/cancel dello stesso ordine;
- nessun merge tra client, lati, prezzi o partizioni diversi.

**Step 4: verificare RED**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_spoofing_metrics.py -k 'execution_cluster or fragmented_fill' -vv
```
Expected: FAIL perché campi e clustering non esistono ancora; non accettare errori di import o fixture come RED valido.

### Task 2: Introdurre il modello cluster puro e testabile

**Objective:** isolare identità, regole di merge e aggregazione quantitativa dalla logica WMSCI.

**Files:**
- Create: `src/spoofing_detection/lob/execution_clusters.py`
- Create: `tests/lob/test_execution_clusters.py`

**Step 1: definire strutture pure**

Implementare dataclass/funzioni senza dipendenze dal filesystem:

```python
@dataclass
class PendingExecutionCluster:
    partition_id: str
    order_id: str
    client_original_id: str | None
    side: str
    first_sort_index: int
    last_sort_index: int
    start_ts: datetime
    end_ts: datetime
    weighted_price_notional: float = 0.0
    fill_qty: float = 0.0
    child_fill_count: int = 0


def can_extend_execution_cluster(
    cluster: PendingExecutionCluster,
    fill: Mapping[str, Any],
    *,
    max_gap_ms: int,
    intervening_lifecycle_break: bool,
) -> bool:
    ...


def finalize_execution_cluster(
    cluster: PendingExecutionCluster,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ...
```

**Step 2: testare casi limite**

- timestamp mancanti;
- quantità zero/negative;
- prezzo mancante;
- fill terminale;
- esatta uguaglianza alla soglia;
- stabilità dell’ID;
- somma quantità e VWAP con `pytest.approx`.

**Step 3: verificare RED, implementare minimo, verificare GREEN**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_execution_clusters.py -vv
```
Expected dopo implementazione: tutti i test del modulo PASS.

### Task 3: Integrare i cluster nello streaming LOB

**Objective:** produrre un solo execution row per cluster mantenendo raw provenance e corretti snapshot temporali.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: estendere `SpoofingMetricResult`**

Aggiungere:
- `execution_cluster_members`;
- `execution_cancel_candidates`.

**Step 2: modificare `_stream_metric_inputs`**

- Bufferizzare i fill passivi eleggibili per ordine.
- Conservare dal primo child: pre-book summary, candidate rows, queue snapshot e attributi client/firm.
- Aggiornare quantità, VWAP e ultimo timestamp con i child successivi.
- Tracciare eventi lifecycle che spezzano il cluster.
- Finalizzare i cluster ai boundary definiti nella sezione 2.2.
- Non cambiare la ricostruzione raw in `panel.py`.

**Step 3: modificare `attach_sci_window_metrics`**

- lookup pre-state su `cluster_first_sort_index`;
- finestra post/cancel da `cluster_end_ts`;
- `fill_qty` cluster-level;
- quality flags se prezzo/timestamp/quantità non riconciliabili.

**Step 4: verificare invarianti**

```python
assert members.group_by("execution_cluster_id").agg(
    pl.col("child_fill_qty").sum().alias("member_fill_qty")
).join(execution_metrics, on="execution_cluster_id").select(
    pl.col("member_fill_qty").sub("fill_qty").abs().max()
).item() < 1e-9
```

**Step 5: test GREEN**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_execution_clusters.py tests/lob/test_spoofing_metrics.py -vv
```
Expected: PASS; i vecchi test devono essere aggiornati solo dove il contratto cambia intenzionalmente.

### Task 4: Deduplicare deterministicamente le cancellazioni

**Objective:** impedire che la stessa cancellazione alimenti più cluster canonici.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: aggiungere test failing**

Costruire due cluster compatibili con la stessa cancellazione e verificare:
- due righe candidate auditabili;
- una sola `assigned_flag == True`;
- assegnazione al cluster con end timestamp più recente;
- tie-break deterministico;
- quantità cancellata conteggiata una sola volta nelle metriche aggregate.

**Step 2: implementare funzione pura di assegnazione**

```python
def assign_cancellations_to_clusters(
    candidate_links: pl.DataFrame,
) -> pl.DataFrame:
    """Assign each unique cancellation to at most one preceding cluster."""
    ...
```

Usare come chiave cancellazione almeno `partition_id`, `cancel_sort_index`, `candidate_order_id`; non usare solo `ORDERID` perché un ID potrebbe essere riutilizzato in partizioni differenti.

**Step 3: calcolare WMSCI solo dai link assegnati**

- Candidate posture pre-fill può includere ordini non cancellati.
- `matched_spoofing_event` è vero solo se esiste almeno un link assegnato.
- `matched_cancellation_count`, quantità e frazioni devono essere cluster-level e non duplicati.

**Step 4: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_spoofing_metrics.py -k 'cancel or match or cluster' -vv
```
Expected: PASS.

### Task 5: Propagare la nuova unità a client-session, alert e report

**Objective:** rendere denominatori, ripetizione e workload coerenti con cluster unici.

**Files:**
- Modify: `src/spoofing_detection/lob/client_session_features.py`
- Modify: `src/spoofing_detection/lob/alert_objects.py`
- Modify: `src/spoofing_detection/lob/spoofing_metric_report.py`
- Modify: `tests/lob/test_client_session_features.py`
- Modify: `tests/lob/test_alert_objects.py`
- Modify: `tests/lob/test_spoofing_metric_report.py`

**Step 1: test failing**

Verificare che cinque child fill appartenenti allo stesso cluster producano:
- `execution_count == 1`;
- `matched_event_count == 1` se c’è un match assegnato;
- una sola osservazione nei massimi/medie WMSCI;
- nessun aumento artificiale di MES/MCPS/alert score.

**Step 2: implementare aggregazioni su `execution_cluster_id`**

- Assert uniqueness prima del group-by.
- Rinominare etichette user-facing da “fill events” a “execution clusters” dove necessario.
- Mantenere un conteggio separato `raw_fill_message_count` per audit, senza usarlo nello score.

**Step 3: test mirati**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_client_session_features.py \
  tests/lob/test_alert_objects.py \
  tests/lob/test_spoofing_metric_report.py -vv
```
Expected: PASS.

### Task 6: Esporre configurazione, output e provenance nella CLI

**Objective:** rendere il clustering riproducibile e auto-descrivente.

**Files:**
- Modify: `configs/spoofing_detection_parameters.json`
- Modify: `scripts/compute_spoofing_metrics.py`
- Modify: `tests/lob/test_compute_spoofing_metrics_cli.py`
- Modify: `docs/spoofing_llm_review_workflow.md`
- Modify: `docs/spoofing_production_readiness.md`

**Step 1: aggiungere parametro**

```json
{
  "event_metrics": {
    "execution_cluster_max_gap_ms": 100
  }
}
```

Validare intero non negativo; `0` significa non unire fill con timestamp differenti.

**Step 2: scrivere nuovi artefatti**

Aggiornare `run()` per emettere Parquet/CSV di membri e link; includere in `metadata.json`:
- versione schema;
- parametro cluster;
- regola di boundary;
- regola di assegnazione cancellazioni;
- conteggi raw fill / cluster / cluster matched / cancellazioni uniche;
- path input/config/kernel;
- comando eseguito;
- hash SHA-256 degli input e del config, calcolato dal programma senza stampare segreti;
- timestamp UTC di creazione.

**Step 3: aggiungere test CLI**

Verificare file presenti, metadati non null e riconciliazione conteggi.

**Step 4: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_compute_spoofing_metrics_cli.py -vv
```
Expected: PASS.

### Task 7: Aggiungere un audit di sensibilità del clustering

**Objective:** evitare che la soglia di 100 ms sia presentata come scelta scientificamente inevitabile.

**Files:**
- Create: `scripts/audit_execution_clustering.py`
- Create: `tests/lob/test_audit_execution_clustering.py`

**Step 1: implementare CLI read-only sugli input metrici**

Argomenti:
```text
--metrics-root
--gaps-ms 25 50 100 250 1000
--output-dir
```

Output:
- `execution_cluster_sensitivity.csv` con fill count, cluster count, matched cluster count, cancellazioni assegnate, top-client ranking, max/quantili WMSCI;
- `execution_cluster_gap_distribution.csv` con quantili dei gap same-order;
- `metadata.json` con fonti e comando.

**Step 2: testare monotonicità e riproducibilità**

- cluster count non aumenta all’aumentare della soglia;
- somma delle quantità child resta invariata;
- stessa configurazione produce stesso output ordinato.

**Step 3: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_audit_execution_clustering.py -vv
```
Expected: PASS.

### Task 8: Arricchire i dossier con evidenza di esecuzione e spiegazioni alternative

**Objective:** fornire al modello dati sufficienti per distinguere matched withdrawal da liquidity provision/inventory management.

**Files:**
- Modify: `scripts/build_spoofing_event_dossier.py`
- Modify: `tests/lob/test_spoofing_event_dossier.py`

**Step 1: cambiare identità evento**

- Lookup primario tramite `execution_cluster_id`/`review_event_id`.
- Includere `execution_cluster_members` nel dossier.
- Rifiutare dossier ambiguo se il cluster ID non è unico.

**Step 2: aggiungere sezioni deterministiche**

```json
{
  "execution_cluster": {
    "child_fill_count": 5,
    "total_fill_qty": 25000,
    "start_ts": "...",
    "end_ts": "...",
    "vwap": 0.0223
  },
  "candidate_order_execution_risk": [],
  "client_inventory_context": {},
  "same_client_bilateral_activity": {},
  "cancellation_assignment": {},
  "data_quality": {}
}
```

Per ogni candidate order calcolare, quando osservabile:
- posizione/rank pre-cluster e distanza dal touch;
- quantità iniziale osservata, eseguita, cancellata e residua;
- fill fraction e cancellation fraction con denominatore esplicito;
- durata e timestamp lifecycle;
- se era al best/near-touch/deep;
- se la cancellazione è assegnata al cluster o solo candidata.

Per il client, nel contesto temporale configurato:
- buy quantity, sell quantity e signed net inventory flow;
- numero/quantità di ordini bilaterali;
- eventuale ordine successivo di segno opposto compatibile con rebalancing;
- caveat che questo è comportamento “market-making-like”, non prova dello status di market maker.

**Step 3: aggiungere test**

- Caso candidate parzialmente eseguita: le frazioni si riconciliano.
- Caso fill cluster: quantità dossier = somma membri.
- Caso client ID mancante/`0`: nessuna attribuzione forte.
- Caso cancellazione concorrente non assegnata: mostrata come alternativa, esclusa dalle metriche canoniche.

**Step 4: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_spoofing_event_dossier.py -vv
```
Expected: PASS.

### Task 9: Sostituire il contratto LLM con una review evidence-layered

**Objective:** impedire che il modello trasformi uno screening meccanico in una conclusione di spoofing.

**Files:**
- Modify: `prompts/spoofing_surveillance_analyst.md`
- Modify: `scripts/analyze_spoofing_event_with_llm.py`
- Modify: `tests/lob/test_spoofing_event_llm_analysis.py`
- Modify: `docs/spoofing_llm_review_workflow.md`

**Step 1: rendere obbligatorie sezioni nell’ordine seguente**

1. `Observed facts`
2. `Data quality and provenance`
3. `Mechanical matched-withdrawal signal`
4. `Execution risk of the withdrawn order`
5. `Position relative to the touch`
6. `Timing and order duration`
7. `Price response and economic benefit`
8. `Bilateral activity and inventory context`
9. `Alternative legitimate explanations`
10. `Evidence against the spoofing hypothesis`
11. `Surveillance priority and confidence`
12. `Intent limitation`

**Step 2: imporre categorie prudenti**

Il modello deve scegliere e motivare una delle categorie:
- `mechanical_matched_withdrawal_signal`;
- `economically_consistent_with_spoofing`;
- `compatible_with_legitimate_liquidity_provision`;
- `requires_human_review`.

La frase finale deve dichiarare che l’intento non è stabilito dai dati disponibili. Vietare equivalenze deterministiche come:
- deep order = spoofing;
- best-level order = market maker;
- alto WMSCI = manipolazione provata;
- review LLM = ground truth.

**Step 3: aggiornare il prompt builder**

Includere esplicitamente:
- schema/versione dossier;
- cluster ID e child count;
- campi mancanti;
- associazione cancellazione canonica;
- richiesta di citare solo valori presenti nel dossier.

**Step 4: test**

I test devono verificare presenza di tutte le sezioni, categorie, limitazione d’intento e contesto cluster; non testare la qualità semantica tramite semplici substring soltanto dove una struttura JSON/Markdown può essere validata deterministicamente.

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_spoofing_event_llm_analysis.py -vv
```
Expected: PASS.

### Task 10: Rendere batch e dashboard cluster-aware e validabili

**Objective:** generare una review per cluster e mostrare sia il cluster sia i child raw senza duplicare l’episodio.

**Files:**
- Modify: `scripts/batch_analyze_spoofing_events_with_llm.py`
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `tests/lob/test_spoofing_event_review_dashboard.py`
- Create: `scripts/validate_spoofing_event_reviews.py`
- Create: `tests/lob/test_validate_spoofing_event_reviews.py`

**Step 1: batch**

- Iterare su `execution_cluster_id` unici.
- Una directory per cluster.
- Non riutilizzare review `S...` message-level.
- Checkpoint atomico dopo ogni cluster.
- Manifest con status `complete`, `failed`, `skipped`, retry count, modello/provider, prompt hash, dossier hash, timestamps e messaggio errore sanificato.
- Non salvare token o header API.

**Step 2: dashboard**

- Riga principale = execution cluster.
- Sezione espandibile “Raw child fills” con i cinque messaggi del caso di regressione.
- Candidate cancellation mostrata una sola volta; badge per `assigned` vs `competing/unassigned`.
- Etichette cluster-level in tabelle, contatori e filtri.
- `dashboard_refreshed_at_utc` obbligatorio e non null.
- Preservare sezioni investigative esistenti: Event selection, LOB views, event-window tables, annotations/review panels e highlighted rows.

**Step 3: validatore review**

La CLI deve verificare:
- set atteso cluster = set dossier = set review complete;
- nessun duplicato;
- schema/versione e sezioni obbligatorie;
- hash dossier/prompt coerenti;
- nessuna review stale incorporata;
- error manifest completo;
- nessun segreto nei file;
- dashboard metadata coerente.

Exit code non zero se manca anche un solo cluster richiesto, salvo allowlist esplicita versionata.

**Step 4: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_spoofing_event_review_dashboard.py \
  tests/lob/test_validate_spoofing_event_reviews.py -vv
```
Expected: PASS.

### Task 11: Eseguire review scientifica del codice prima dei batch costosi

**Objective:** bloccare rigenerazioni se unità, finestre o denominatori non sono corretti.

**Files:** nessun file obbligatorio; correggere solo finding concreti nei file dei Task 2–10.

**Step 1: scientific/specification review**

Verificare adversarialmente:
- conservazione quantità;
- boundary di partizione/sessione;
- nessun look-ahead nel pre-state;
- cancellation delay da cluster end;
- candidate age da cluster start;
- univocità cancellazione assegnata;
- sensibilità al gap;
- client ID mancanti;
- nessuna inferenza d’intento.

**Step 2: code-quality/regression review**

Verificare:
- deterministic ordering;
- NaN/Inf e dtype;
- output vuoti;
- timestamp duplicati;
- API backward compatibility intenzionale;
- nessuna duplicazione di logica tra dossier/dashboard.

**Step 3: suite completa**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest -q
```
Expected: exit code `0`; registrare il conteggio reale, senza anticiparlo nel paper.

### Task 12: Rigenerare metriche in nuove directory e fare il sensitivity audit

**Objective:** produrre dati cluster-level per RISANAMENTO, NEXI e FERRARI senza sovrascrivere il run precedente.

**Files created:**
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/RISANAMENTO_.../`
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/NEXI_.../`
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/FERRARI_.../`
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/clustering_sensitivity/`

**Step 1: usare gli stessi input/kernel/parametri del run canonico**

Ricavare i comandi dai `metadata.json` del run 20260714; cambiare solo:
- output directory;
- schema clustering;
- parametro `execution_cluster_max_gap_ms`.

**Step 2: eseguire i tre run**

Usare `/home/danielemdn/miniconda3/envs/main/bin/python scripts/compute_spoofing_metrics.py ...` con gli argomenti completi ricavati e salvare il comando in metadata. Non usare wildcard che possano selezionare un kernel errato.

Expected:
- exit code `0` per ogni strumento;
- artefatti cluster/member/link presenti;
- nessun overwrite del run precedente.

**Step 3: eseguire sensitivity grid**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python scripts/audit_execution_clustering.py \
  --metrics-root outputs/spoofing_metrics/<NEW_RUN_ID> \
  --gaps-ms 25 50 100 250 1000 \
  --output-dir outputs/spoofing_metrics/<NEW_RUN_ID>/clustering_sensitivity
```
Expected: CSV/metadata completi e invarianti quantità soddisfatti.

**Step 4: gate RISANAMENTO `S101248`**

Verificare programmaticamente:
- un solo cluster contenente gli indici `101248`, `101251`, `101254`, `101257`, `101260`;
- `child_fill_count = 5`;
- `fill_qty = 25000`;
- cancellazione candidata assegnata una sola volta;
- withdrawal-to-fill ratio `8.7908` se la quantità cancellata validata è `219770`;
- nessuna delle quattro sibling row residue compare come matched event autonomo.

Se uno di questi gate fallisce, fermarsi: non avviare review LLM e non aggiornare i risultati del paper.

### Task 13: Rigenerare production readiness e workload cluster-level

**Objective:** riallineare alert e conteggi usati nel paper.

**Files created:**
- New: `outputs/spoofing_production/<NEW_RUN_ID>/...`

**Step 1: eseguire readiness sui tre run nuovi**

Usare `scripts/run_spoofing_production_readiness.py` con path espliciti dei nuovi `execution_metrics.parquet`.

**Step 2: riconciliare**

Per ogni strumento:
- `execution_count == n_unique(execution_cluster_id)`;
- `matched_event_count == n_unique(cluster matched)`;
- somma eventi client = totale per client non null secondo il contratto;
- alert count derivato esclusivamente da dati cluster-level.

**Step 3: salvare report di validazione**

Creare `validation_summary.json`/`.md` nella directory run con:
- check, expected, observed, pass/fail;
- path e hash fonti;
- test command e result;
- sensitivity findings.

### Task 14: Generare tutti i dossier e le review RISANAMENTO

**Objective:** ottenere copertura completa del nuovo set cluster-level, senza riusare le 136 review message-level precedenti.

**Files created:**
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/RISANAMENTO_.../event_review/`

**Step 1: generare dossier per tutti i matched cluster**

- Un dossier per ogni `execution_cluster_id` matched.
- Validare immediatamente schema e riconciliazione prima delle chiamate LLM.

**Step 2: dry-run del batch**

Verificare manifest atteso, provider/modello e numero di cluster senza inviare richieste.

**Step 3: batch LLM con resume**

Eseguire il batch reale solo dopo gate dei Task 11–13. Usare retry limitato e backoff; preservare errori per cluster senza bloccare la provenienza degli altri.

**Step 4: validare e colmare solo failure reali**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python scripts/validate_spoofing_event_reviews.py \
  --metrics-dir outputs/spoofing_metrics/<NEW_RUN_ID>/RISANAMENTO_... \
  --review-dir outputs/spoofing_metrics/<NEW_RUN_ID>/RISANAMENTO_.../event_review
```
Expected prima della dashboard finale: exit code `0`, copertura `100%` del set cluster atteso e zero review stale.

**Step 5: rigenerare dashboard**

Expected:
- contatori cluster-level;
- review complete incorporate;
- `dashboard_refreshed_at_utc` non null;
- child fills e cancellation assignment visibili.

### Task 15: Creare un generatore riproducibile per tabelle e macro del paper

**Objective:** eliminare la trascrizione manuale dei numeri dai risultati.

**Files:**
- Create: `scripts/build_spoofing_paper_results.py`
- Create: `tests/lob/test_build_spoofing_paper_results.py`
- Create/generated: `paper/generated/spoofing_results_macros.tex`
- Create/generated: `paper/generated/empirical_spoofing_diagnostics_rows.tex`
- Create/generated: `paper/generated/top_client_results_rows.tex`
- Create/generated: `paper/generated/spoofing_results_metadata.json`

**Step 1: test failing con piccoli input sintetici**

Verificare:
- conteggi da cluster unici;
- percentuali con denominatore matched cluster;
- ordinamento strumenti/client deterministico;
- escaping LaTeX;
- valori mancanti resi come `--` e non zero;
- metadata con fonti/hash/comando.

**Step 2: implementare CLI**

Argomenti espliciti:
```text
--risanamento-metrics-dir
--nexi-metrics-dir
--ferrari-metrics-dir
--risanamento-readiness-dir
--nexi-readiness-dir
--ferrari-readiness-dir
--sensitivity-csv
--output-dir paper/generated
```

**Step 3: usare macro per numeri richiamati nella prosa**

Generare macro per:
- execution cluster, matched cluster, client e alert count;
- max WMSCI e max W/F;
- FPM/REV shares;
- top-client counts/means/max;
- estremi del sensitivity range, se materialmente differenti.

**Step 4: test**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_build_spoofing_paper_results.py -vv
```
Expected: PASS.

### Task 16: Aggiornare definizioni e linguaggio scientifico del manoscritto

**Objective:** allineare teoria, notazione e claims alla nuova unità e ai limiti evidenziari.

**Files:**
- Modify: `paper/spoofing.tex`

**Manuscript edits — tutte le occorrenze visibili da aggiornare:**

1. `paper/spoofing.tex:65-78` — abstract/contributo: chiarire screening esplorativo, cluster-level execution e non-identificazione dell’intento.
2. `paper/spoofing.tex:120-188` — confronto market maker/spoofer: trasformare affermazioni assolute (“purely directional”, “rarely”, “exact millisecond”) in meccanismi canonici/tendenze; esplicitare che best-level placement e partial execution supportano spiegazioni alternative.
3. `paper/spoofing.tex:403-424` — timeline: “Execution” diventa “Execution cluster”; definire start/end e trattamento dei child fill.
4. `paper/spoofing.tex:760-915` — definizioni MSCI/WMSCI:
   - introdurre set dei child fill `\mathcal{M}_e`;
   - definire `Q_e = \sum_{m\in\mathcal{M}_e} q_m`;
   - pre-state prima del primo child e cancel delay dall’ultimo child;
   - definire associazione univoca delle cancellazioni;
   - sostituire “one execution/after a small fill” con terminologia cluster coerente.
5. `paper/spoofing.tex:992-1061` — MCPS e figura TikZ: denominatore = execution cluster unici; asse/caption “execution cluster index”; evitare che punti rossi/blu suggeriscano ground truth osservata se la figura è solo illustrativa.
6. `paper/spoofing.tex:1081-1158` — price-response ed event reconstruction: aggiornare `t_e` a start/end cluster, ordine delle operazioni, dedup cancellazioni e audit raw→cluster.
7. `paper/spoofing.tex:1160-1289` — Results: sostituire tutti i numeri hard-coded con macro/tabelle generate dai nuovi output; descrivere sensitivity della soglia; distinguere raw fill messages, execution cluster, matched cluster e client sessions; eliminare claims basati sui vecchi `219/573/364`, massimi o top-client counts finché non rigenerati.
8. `paper/spoofing.tex:1180-1245` — tabelle empiriche: usare `\input{generated/..._rows.tex}` e macro; aggiornare header “Executions” a “Execution clusters” e “Matched events” a “Matched clusters”.
9. `paper/spoofing.tex:1265-1289` — alert prioritization: definire MES/score con cluster unici e mantenere esplicito che gli alert sono workload, non episodi provati.
10. `paper/spoofing.tex:1308-1319` — anche se commentato, non è visibile nel PDF e non richiede modifica per il build; se viene riattivato in futuro, andrà riscritto perché contiene equivalenze deterministiche. Non riattivarlo in questo lavoro.

**Generator-level changes separati:**

- Non modificare manualmente immagini raster/PDF: non esistono include grafici attivi per questi risultati.
- Aggiornare solo le figure TikZ inline in `paper/spoofing.tex` dove l’etichetta “execution” deve diventare “execution cluster”.
- Tutti i numeri empirici visibili devono provenire da `paper/generated/` creato nel Task 15.

**Step 1: aggiungere test/manual checklist di coerenza notazionale**

Search obbligatorie dopo edit:
```bash
rg -n "after one execution|after a small fill|Each point is one execution|Matched events|Instrument & Executions" paper/spoofing.tex
```
Expected: nessuna occorrenza attiva obsoleta; commenti possono restare solo se chiaramente inattivi e non contraddicono testo riattivabile.

**Step 2: verificare che ogni numero di risultati sia generato**

Search:
```bash
rg -n "219|573|364|93\.09|18,200|44\.7|62\.1|58\.0" paper/spoofing.tex
```
Expected: nessun vecchio valore empirico hard-coded nel testo attivo.

### Task 17: Rigenerare artefatti paper e compilare

**Objective:** produrre un PDF leggibile e coerente con gli output validati.

**Files created/updated:**
- `paper/generated/*.tex`
- `paper/generated/spoofing_results_metadata.json`
- Build artifacts under `paper/` (non aggiungerli a Git se ignorati)

**Step 1: generare macro e righe tabella**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python scripts/build_spoofing_paper_results.py \
  --risanamento-metrics-dir <EXACT_NEW_PATH> \
  --nexi-metrics-dir <EXACT_NEW_PATH> \
  --ferrari-metrics-dir <EXACT_NEW_PATH> \
  --risanamento-readiness-dir <EXACT_NEW_PATH> \
  --nexi-readiness-dir <EXACT_NEW_PATH> \
  --ferrari-readiness-dir <EXACT_NEW_PATH> \
  --sensitivity-csv <EXACT_NEW_PATH>/execution_cluster_sensitivity.csv \
  --output-dir paper/generated
```
Expected: exit code `0`; metadata e quattro artefatti generati presenti.

**Step 2: compilare**

Preferito:
```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error spoofing.tex
```

Se `latexmk` non è disponibile:
```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error spoofing.tex
bibtex spoofing
pdflatex -interaction=nonstopmode -halt-on-error spoofing.tex
pdflatex -interaction=nonstopmode -halt-on-error spoofing.tex
```
Expected: exit code `0`, `spoofing.pdf` prodotto, nessun undefined reference/citation.

**Step 3: ispezione log e visiva**

- Cercare `Undefined`, `Citation`, `Reference`, `Overfull`, `Underfull`, `Error` in `paper/spoofing.log`.
- Ispezionare le pagine con timeline, MCPS, tabelle risultati e alert score nel PDF compilato.
- Verificare font leggibili alla dimensione finale, colonne non tagliate e caption coerenti.

### Task 18: Validazione finale end-to-end e report

**Objective:** dimostrare completezza, correttezza e provenance senza trasformare risultati esplorativi in claims confermativi.

**Files created:**
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/validation/final_validation_report.md`
- New: `outputs/spoofing_metrics/<NEW_RUN_ID>/validation/final_validation_report.json`

**Step 1: software verification**

Run:
```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest -q
```
Expected: exit code `0`.

**Step 2: scientific invariants**

Per ciascuno strumento:
- somma child fill per cluster = `fill_qty` cluster;
- ogni child fill appartiene al massimo a un cluster;
- ogni cancellazione è assegnata al massimo a un cluster;
- pre-state precede il primo child;
- cancel delay è non negativo e parte dal cluster end;
- nessun cluster attraversa partizioni/sessioni;
- nessun NaN/Inf nelle metriche finite attese;
- conteggi metadata = righe/n_unique effettivi;
- sensitivity table completa.

**Step 3: review completeness/provenance**

RISANAMENTO:
- set cluster matched = set dossier = set review valida = set dashboard;
- zero review message-level stale;
- zero errori non risolti oppure elenco esplicito che impedisce il claim “complete”;
- prompt hash, dossier hash, modello/provider e timestamp presenti;
- nessun segreto.

**Step 4: paper provenance**

- hash/path in `paper/generated/spoofing_results_metadata.json` corrispondono agli output validati;
- tabelle e prosa usano macro generate;
- PDF compila senza errori;
- risultati descritti come esplorativi;
- se sensitivity cambia sostanzialmente ranking/conteggi, il testo lo dichiara.

**Step 5: due review pass finali**

1. **Scientific/specification:** leakage, unità, denominatori, boundary, sensibilità, alternative legittime, limitazione d’intento.
2. **Code-quality/regression:** diff scope, determinismo, schema, error handling, test, output paths, nessun overwrite.

Non dichiarare completamento se anche un solo gate obbligatorio è rosso.

## 5. File likely to change

### Core scientific code
- `src/spoofing_detection/lob/execution_clusters.py` (new)
- `src/spoofing_detection/lob/spoofing_metrics.py`
- `src/spoofing_detection/lob/client_session_features.py`
- `src/spoofing_detection/lob/alert_objects.py`
- `src/spoofing_detection/lob/spoofing_metric_report.py`

### Pipeline, dossier, LLM e dashboard
- `configs/spoofing_detection_parameters.json`
- `scripts/compute_spoofing_metrics.py`
- `scripts/audit_execution_clustering.py` (new)
- `scripts/build_spoofing_event_dossier.py`
- `scripts/analyze_spoofing_event_with_llm.py`
- `scripts/batch_analyze_spoofing_events_with_llm.py`
- `scripts/build_spoofing_event_review_dashboard.py`
- `scripts/validate_spoofing_event_reviews.py` (new)
- `prompts/spoofing_surveillance_analyst.md`

### Paper e artefatti generati
- `scripts/build_spoofing_paper_results.py` (new)
- `paper/spoofing.tex`
- `paper/generated/spoofing_results_macros.tex` (generated)
- `paper/generated/empirical_spoofing_diagnostics_rows.tex` (generated)
- `paper/generated/top_client_results_rows.tex` (generated)
- `paper/generated/spoofing_results_metadata.json` (generated)

### Tests
- `tests/lob/test_execution_clusters.py` (new)
- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_client_session_features.py`
- `tests/lob/test_alert_objects.py`
- `tests/lob/test_spoofing_metric_report.py`
- `tests/lob/test_compute_spoofing_metrics_cli.py`
- `tests/lob/test_audit_execution_clustering.py` (new)
- `tests/lob/test_spoofing_event_dossier.py`
- `tests/lob/test_spoofing_event_llm_analysis.py`
- `tests/lob/test_spoofing_event_review_dashboard.py`
- `tests/lob/test_validate_spoofing_event_reviews.py` (new)
- `tests/lob/test_build_spoofing_paper_results.py` (new)

### Documentation
- `docs/spoofing_llm_review_workflow.md`
- `docs/spoofing_production_readiness.md`

## 6. Acceptance criteria

1. Il caso `S101248` è rappresentato da un solo cluster da `25000` con cinque child fill auditabili.
2. La cancellazione associata compare una sola volta nelle metriche canoniche.
3. Tutte le quantità cluster si riconciliano con i child raw entro tolleranza numerica dichiarata.
4. Client-session e alert usano cluster unici, non messaggi child.
5. I dossier espongono fill fraction della candidate order, rank/touch, contesto bilaterale, inventory flow, qualità dati e associazione cancellazione.
6. Le review separano fatti, screening, evidenza economica, alternative legittime e limiti d’intento.
7. Tutti i matched cluster RISANAMENTO hanno dossier e review valide nel nuovo run; la dashboard non incorpora review stale.
8. `dashboard_refreshed_at_utc` è valorizzato.
9. La sensitivity grid è prodotta e discussa se materialmente rilevante.
10. Le tabelle/numeri del paper derivano dagli output validati, non da trascrizione manuale.
11. Teoria, equazioni, caption, figure TikZ e Results usano coerentemente “execution cluster”.
12. La suite pytest e la compilazione LaTeX terminano con exit code `0`.
13. Nessun input raw o output canonico precedente viene sovrascritto.
14. Nessun segreto viene salvato nei manifest, dossier, review o log.

## 7. Rischi, trade-off e decisioni aperte

- **Soglia temporale:** `100` ms è operativa. Il risultato deve essere accompagnato da sensitivity; se instabile, il paper non deve presentare un unico conteggio come robusto.
- **Partial fills distanti:** una soglia corta può separare fill dello stesso ordine economicamente collegati; una soglia lunga può fondere episodi distinti. Il report dei gap e la lifecycle boundary sono indispensabili.
- **Regola latest-cluster:** assegna univocamente una cancellazione ma non dimostra causalità. L’artefatto candidate-links deve conservare tutte le alternative.
- **Client ID `0`/mancante:** non trattarlo automaticamente come un singolo agente economico; segnalarlo come limite di attribuzione.
- **Vecchie review:** i 136 output precedenti sono message-level e non vanno conteggiati come copertura del nuovo schema.
- **Costi/variabilità LLM:** il batch va eseguito solo dopo i gate deterministici; modello, prompt e dossier hash devono essere fissati per rendere confrontabili le review.
- **Risultati cross-instrument:** anche se le review richieste sono RISANAMENTO, il paper confronta tre strumenti; tutte e tre le metriche devono essere rigenerate con la stessa unità prima di aggiornare le tabelle.
- **Scope paper:** non introdurre nuovi claims causali o classificatori di market maker. Il lavoro aggiorna unità, provenance, prudenza inferenziale e numeri validati.
- **Working tree sporco:** durante l’implementazione usare patch localizzate e stage selettivo solo se l’utente autorizza commit; non includere file non correlati.

## 8. Execution handoff

Il piano va eseguito in ordine. I Task 1–11 costituiscono il gate software/scientifico; i batch e la riscrittura dei risultati (Task 12–18) sono bloccati finché tutti i test e gli invarianti del gate non passano. Se si usa delegazione, assegnare file ownership non sovrapposta e sottoporre ogni task a review di conformità scientifica prima della review di qualità del codice.
