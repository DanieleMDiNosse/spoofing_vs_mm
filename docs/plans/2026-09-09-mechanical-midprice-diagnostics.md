# Componente meccanica del midprice — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Assegnare file disgiunti; il replay eventualmente condiviso con il piano 2 ha un solo proprietario. Non implementare prima di autorizzazione esplicita.

**Goal:** Quantificare quanto delle diagnostiche FPM e reversione sia spiegato aritmeticamente dalla presenza degli ordini candidati nelle migliori quotazioni osservate, mantenendo separato il movimento del book residuo.

**Architecture:** Riprodurre il replay canonico senza sopprimere alcun messaggio. Nei soli stati necessari calcolare due viste: book completo e medesimo book senza un insieme congelato di lifecycle candidati. Decomporre FPM/REV usando gli stessi stati e pesi della baseline, producendo sidecar diagnostici senza cambiare il detector.

**Tech Stack:** Python 3.11 dell'ambiente di progetto, Polars, libreria standard e pytest; Matplotlib per eventuali figure statiche. Nessuna nuova dipendenza.

---

## 1. Stato e decisioni di perimetro

- **Stato:** piano da approvare, nessuna implementazione o analisi empirica avviata.
- Questo è il **punto 3** della discussione sull'attendibilità; il piano [controlli empirici end-to-end](2026-09-09-empirical-negative-controls.md) è separato.
- Si costruisce una **decomposizione descrittiva del book osservato**, non un mercato simulato senza il candidato e non una stima di causalità.
- Il comportamento osservato degli altri partecipanti resta invariato. Le loro reazioni possono dipendere dal candidato: il book residuo non è un controfattuale causale.
- Non cambiare soglie, FPM/REV canonici, gate strict, WMSCI/MCPS, membership degli episodi, ranking, dashboard storiche o paper.
- Non rinominare il risultato residuo «movimento degli altri operatori»: restano anche ordini non candidati dell'attore focale. Il risultato è «midprice del book esclusi i lifecycle candidati».
- Gli episodi restano l'unità principale; cluster e cancellazioni sono dettagli espliciti. Passive/aggressive sono separati nelle diagnostiche, mentre gli episodi misti sono contati una sola volta nei totali.
- Nessun commit, push, aggiornamento di `LATEST` o promozione paper senza richiesta.

## 2. Punti di integrazione verificati

| Fonte | Contratto da preservare |
|---|---|
| `src/spoofing_detection/lob/models.py:9–25` | `ActiveOrder`, quantità visibile e `first_seen_sort_index` |
| `src/spoofing_detection/lob/panel.py:71–127` | `book_summary`, livelli da ordini attivi e midprice da best bid/ask |
| `src/spoofing_detection/lob/panel.py:400–499` | Gestione ordini resting/non-resting e residuali aggressivi |
| `src/spoofing_detection/lob/spoofing_metrics.py:989–1198` | Ordine di normalizzazione, osservazione pre-evento, applicazione e flush |
| `src/spoofing_detection/lob/spoofing_metrics.py:1392–1406` | Segno `+1` per execution side `ask`, `-1` per `bid` |
| `src/spoofing_detection/lob/spoofing_metrics.py:1482–1543` | FPM da stato esatto post-ultima-pubblicazione candidata a pre-primo-fill |
| `src/spoofing_detection/lob/spoofing_metrics.py:1784–1876` | Reversione dalla pre-cancellazione all'orizzonte osservato, pesi quantità/ritardo |
| `src/spoofing_detection/lob/candidate_episodes.py:163–176, 229–262` | Chiavi lifecycle, FPM ponderato per quantità eseguita e REV per cancellazioni uniche |
| `docs/keep_cols_data_dictionary.md:72–77, 95–105` | Namespace di ORDERID, trade ID e caveat sul ruolo passivo/aggressivo |

I numeri di riga sono riferimenti di pianificazione, da ricontrollare sul tree prima dell'implementazione. Gli snapshot top-N o di fine partizione non garantiscono da soli le informazioni necessarie per questa diagnostica.

## 3. Oggetto matematico e convenzioni

### 3.1 Set di esclusione fisso

Per ogni episodio `e`, definire `C_e` come l'unione deduplicata dei lifecycle candidati di **tutti** i cluster membri, non soltanto degli ordini cancellati o dei cluster strict.

Chiave richiesta:

`(partition_id, event_date, order_id, first_seen_sort_index)`.

La partizione deve distinguere strumento e meccanismo di mercato secondo lo schema effettivo. Conservare `actor_key`, livello di identità e lato nei dati di membership e verificarli prima del replay. Non sostituire ORDERID con ORDERPRIORITY e non collegare cancel-replace senza prova.

- Il set è identico in tutti gli stati/confronti dello stesso episodio. Non riselezionare i candidati in base al book o al risultato residuo.
- Un lifecycle non ancora presente o già terminato non rimuove nulla nello stato corrispondente. Un nuovo lifecycle con lo stesso ORDERID non viene escluso per errore.
- La scelta tramite episodio completo è **retrospettiva**: usa membership conosciuta ex-post. Non rappresenta una feature disponibile in tempo reale alla pubblicazione.
- Una variante futura per set del singolo cluster o per tutti gli ordini dell'attore è un'analisi diversa, con nome/policy distinti; esclusa dalla v1.

### 3.2 Due viste dello stesso stato

Sia `B_t` il book attivo osservato al confine canonico `t`, e `B_t^{-C_e}` lo stesso dizionario filtrato per lifecycle, senza modificare `B_t`.

- `m_t = (best_bid_t + best_ask_t) / 2`;
- `m_t_excl = (best_bid_t_excl + best_ask_t_excl) / 2`;
- `g_t = m_t - m_t_excl`: differenza aritmetica di quotazione dovuta alla presenza dei candidati nello stato osservato.

Calcolare i best dalla profondità visibile realmente ricostruita, **non soltanto dai primi N livelli salvati**: dopo l'esclusione il best residuo potrebbe essere più lontano. Top-N è la regola di candidatura, non il limite della ricerca del best residuo.

Applicare la stessa regola di visibilità di `book_summary`. Filtrare gli ordini e riaggregare i livelli, senza sottrarre più volte quantità da snapshot ripetuti. Gli altri ordini allo stesso prezzo rimangono.

Se un lato manca, la profondità residua è ignota, il book è invalido o il clock non è interpretabile, i componenti non valutabili sono `null` con reason. Uno zero è ammesso solo quando il calcolo è osservabile. Conservare quote e flag raw anche quando la decomposizione viene esclusa per qualità.

### 3.3 FPM: stessa baseline post-placement

Convenzione economica già implementata: `d = +1` per una vendita (`execution_side = ask`), `d = -1` per un acquisto (`bid`).

Per ogni cluster `j` dell'episodio:

- `p_j`: indice esatto **post** ultima pubblicazione del set candidato del cluster, già registrato dal detector;
- `f_j_minus`: stato pre-primo-fill già selezionato dal detector.

Definire:

```text
FPM_obs_j  = d * (m[f_j_minus]      - m[p_j])
FPM_excl_j = d * (m_excl[f_j_minus] - m_excl[p_j])
FPM_mech_j = d * (g[f_j_minus]      - g[p_j])
FPM_obs_j  = FPM_excl_j + FPM_mech_j
```

Questa identità usa lo stesso `C_e` nei due estremi. `FPM_obs_j` deve coincidere con `favorable_mid_move_pre_fill` del run di riferimento quando osservabile.

**Distinzione importante:** il salto immediato causato dalla pubblicazione non entra nel FPM corrente, perché la baseline è già post-placement. Un semplice ordine che migliora il best e poi resta immobile può avere un salto iniziale ma FPM nullo. Non cambiare la baseline per aumentare il numero di casi spiegati meccanicamente.

### 3.4 Reversione: stessa cancellazione, stesso orizzonte

Per ogni cancellazione fisica assegnata `c`:

- `c_minus`: stato pre-cancellazione canonico;
- `h_c`: ultimo stato ammesso a/before `cancel_ts + H`, con copertura effettiva fino al target e vincoli di indice/giorno del detector.

```text
REV_obs_c  = d * (m[c_minus]      - m[h_c])
REV_excl_c = d * (m_excl[c_minus] - m_excl[h_c])
REV_mech_c = d * (g[c_minus]      - g[h_c])
REV_obs_c  = REV_excl_c + REV_mech_c
```

Non confondere l'istante dell'ultimo stato selezionato con una quotazione nuova esattamente al target; emettere entrambi i timestamp. Una serie terminata prima del target non offre copertura, anche se il software può tecnicamente propagare l'ultimo valore.

### 3.5 Salti immediati e tratto successivo

Aggiungere come diagnostica distinta gli stati esatti pre/post di ogni pubblicazione osservata e cancellazione candidata; non limitarsi alla prima/ultima quando si vuole parlare di salti delle singole azioni.

```text
INSERT_obs = d * (m[placement_plus] - m[placement_minus])
INSERT_excl = d * (m_excl[placement_plus] - m_excl[placement_minus])
INSERT_mech = d * (g[placement_plus] - g[placement_minus])

REV_immediate_obs = d * (m[c_minus] - m[c_plus])
REV_later_obs     = d * (m[c_plus]  - m[h_c])
REV_obs          = REV_immediate_obs + REV_later_obs
```

Applicare la stessa scomposizione alle viste escluse e alla differenza `g`. Non sommare questi salti per sostituire FPM o REV: possono riguardare clock, stati e periodi sovrapposti diversi. Le modifiche/reload non equivalgono automaticamente a una prima pubblicazione osservata.

Se il trattamento canonico di una riga comprende un flush dei residuali, documentarlo e mantenerlo: il salto pre/post è al confine di replay, non una prova di risposta degli altri operatori. Quote residuo mutate nello stesso confine vanno esposte, non azzerate per costruzione.

### 3.6 Aggregazione ad episodio e branch

Riutilizzare esattamente la membership e i pesi della baseline:

- FPM episodio: media dei componenti cluster con pesi `execution_quantity` e denominatore quantità totale di **tutti** i cluster membri.
- REV episodio: media dei componenti delle **cancellazioni fisiche uniche assegnate**, con peso già registrato `cancel_reversion_weight = attributed_qty * exp(-delay / withdrawal_decay_seconds)`.
- Conservare i pesi esistenti, senza riassegnare il winning cluster nel nuovo modulo.
- Identità additive valide solo sullo stesso insieme di contributi, con gli stessi pesi. Copertura completa richiesta per un componente ufficiale della decomposizione episodio; nessuna rinormalizzazione silenziosa sul sottoinsieme osservato.
- Se il residuo è indisponibile per un solo membro necessario, mantenere l'osservato baseline ma impostare componenti residuo/meccanico episodio a `null`. Esportare conteggi e quote di peso osservate.
- Esporre diagnostiche di branch e totali deduplicati. Le partecipazioni passive/aggressive di un episodio misto non sono due episodi indipendenti.
- `mech / obs` non è necessariamente tra zero e uno e può essere instabile vicino a zero. **Nessuna percentuale di spiegazione come metrica primaria**; conservare componenti firmati in unità prezzo e tick. Il midprice può muoversi di frazioni di tick.

## 4. Dati, replay e prerequisiti

1. Scegliere un run baseline congelato. Il riferimento disponibile al momento del piano è `outputs/spoofing_metrics/20260908_160831_candidate_posture_episodes/`; non sovrascriverlo.
2. Leggere metadati per path raw, ordinamento, normalizzazione, tick, parametri, source hash e artefatti episodio/cluster/cancel. Validare schema e provenienza; non ricostruire i path dai nomi degli strumenti.
3. Conservare la corrispondenza degli indici del run originale. Un sottoinsieme raw non deve essere rinumerato e poi unito ai vecchi `sort_index`.
4. Per un pilot leggere una partizione inizializzata correttamente o riprodurre il suo prefisso; non iniziare il book vuoto trenta secondi prima dell'evento. Liquidità di fondo più vecchia può determinare il best residuo.
5. Programmare le richieste di stato dagli indici già memorizzati in `execution_metrics` e nei link di cancellazione, più pre/post immediati. Se manca un indice necessario, ricalcolarlo con la stessa policy e dimostrare equivalenza; non scegliere il timestamp più vicino.
6. Riprodurre **tutti** gli eventi, inclusi quelli dei candidati. Stop inattivi, ordini marketable/non-resting, trade busts e residuali seguono le regole esistenti; eventuali limiti del replay restano limiti anche del sidecar.
7. Osservare soltanto i punti necessari, riusando stati tra episodi senza memorizzare una copia completa di tutti i book. Non conservare oggetti `ActiveOrder` mutabili per usarli dopo il prossimo evento.
8. I confronti non attraversano il giorno/sessione; lo stato del book viene inizializzato/trasferito soltanto come previsto dal replay canonico, non resettato arbitrariamente per il nuovo modulo.
9. La semantica di ruolo aggressivo non viene certificata da questa diagnostica. Provenienza sconosciuta resta esplicita; nessuna inferenza di intento.

## 5. File previsti e API proposte

I nomi nuovi sono proposte da implementare, non funzioni già disponibili.

| Azione | Percorso | Responsabilità |
|---|---|---|
| Creare | `src/spoofing_detection/lob/mechanical_midprice.py` | Filtri lifecycle, quote delle due viste, decomposizioni e aggregazione |
| Creare/condividere | `src/spoofing_detection/lob/replay_observation.py` | Minima orchestrazione osservabile, estratta dal replay metriche se non già realizzata nel piano 2 |
| Modificare solo se necessario | `src/spoofing_detection/lob/spoofing_metrics.py` | Collegamento equivalente al replay condiviso, nessuna nuova semantica delle metriche |
| Creare | `scripts/compute_mechanical_midprice_diagnostics.py` | Validazione baseline, replay, artifact writer e report |
| Creare | `configs/mechanical_midprice_diagnostics.json` | Manifest baseline, policy di esclusione, selezione pilot, output |
| Creare | `tests/lob/test_mechanical_midprice.py` | Book filtrato, segni, null e identità analitiche |
| Creare | `tests/lob/test_mechanical_midprice_pipeline.py` | Membership, tempi, aggregazione, invariance |
| Creare | `tests/lob/test_mechanical_midprice_cli.py` | Preflight, output e report |
| Creare/condividere | `tests/lob/test_replay_observation.py` | Equivalenza del replay osservabile |

API pure candidate, da definire con type hints e test prima del codice:

- `summarize_excluding_lifecycles(active_orders, *, excluded_lifecycles, partition_id, event_date)`;
- `decompose_signed_mid_change(start, end, *, direction, orientation)`;
- `aggregate_episode_price_components(cluster_components, cancellation_components, membership)`.

`orientation` ammette esclusivamente forward (FPM/inserimento) o reverse (REV); non usare segni impliciti diversi tra funzioni. Ogni riga conserva il tipo di confronto, non solo un numero.

**Coordinamento col piano 2:** se il supporto di replay esiste già, usarlo senza duplicarlo. Se i piani sono eseguiti in parallelo, l'estrazione del replay è un prerequisito seriale con proprietario esclusivo; solo dopo l'equivalenza lavorare sui due moduli diagnostici disgiunti. Nessun refactor generale di `panel.py` o delle regole di mercato.

## 6. Task piccoli con TDD

Per ciascun task: test RED e fallimento per il motivo atteso → minima implementazione → GREEN → diff. Nessun commit automatico.

### Task 1 — Contratto input e calendario delle richieste

**File:** runner/config nuovi e `test_mechanical_midprice_cli.py`.

1. RED per metadata legacy, hash/config incompatibili, lifecycle sconosciuto, indice fuori partizione, output esistente e input vuoto schema-valido.
2. Implementare preflight e `--validate-only` senza scritture.
3. `python -m pytest tests/lob/test_mechanical_midprice_cli.py -q`: GREEN.
4. Costruire tabella richieste con `(episode_id, component_id, state_role, partition_id, sort_index, target_ts)` e set escluso versionato.

### Task 2 — Book filtrato puro

**File:** `mechanical_midprice.py`, `test_mechanical_midprice.py`.

1. RED per ordini allo stesso prezzo, lifecycle riusato, lato residuo assente e input non mutato.
2. Filtrare gli ordini per lifecycle e riusare `book_summary`; non mutare quantità o dizionario originale.
3. `python -m pytest tests/lob/test_mechanical_midprice.py -q`: GREEN.
4. Verificare assenza di esclusioni = identità, candidati non-best = mid invariato quando il touch non cambia, più candidati allo stesso prezzo = una rimozione per ordine.

### Task 3 — Replay ai confini esatti

**File:** supporto condiviso, eventuale integrazione minima, `test_replay_observation.py`, `test_mechanical_midprice_pipeline.py`.

1. RED per timestamp uguali, pending aggressive residual, placement reload non osservato e actor-state sparso.
2. Riusare/estrarre soltanto l'orchestrazione pre/apply/flush/post; osservatore non mutante.
3. `python -m pytest tests/lob/test_replay_observation.py tests/lob/test_reconstruction.py tests/lob/test_episode_timing_contract.py tests/lob/test_mechanical_midprice_pipeline.py -q`: GREEN.
4. Book osservato e tutte le metriche canoniche coincidono con baseline alle stesse chiavi; nessun messaggio rimosso. Un disaccordo blocca la decomposizione empirica.

### Task 4 — Decomposizioni e casi analitici

**File:** `mechanical_midprice.py`, `test_mechanical_midprice.py`.

1. RED per FPM, REV, inserimento e reversione immediata/tardiva, in entrambe le direzioni.
2. Implementare le identità della sezione 3 senza soglie economiche nuove.
3. Eseguire i casi analitici della tabella sotto; confronti numerici con tolleranza dichiarata legata al roundoff, non epsilon nei rapporti.
4. `python -m pytest tests/lob/test_mechanical_midprice.py -q`: GREEN.

### Task 5 — Aggregazione, copertura e membership

**File:** `mechanical_midprice.py`, `test_mechanical_midprice_pipeline.py`.

1. RED per episodio con più cluster, rami misti, cancellazione condivisa, membro senza copertura e set escluso che cambia nel tempo.
2. Aggregare con gli stessi pesi baseline e coverage completa; esporre motivo dei null.
3. `python -m pytest tests/lob/test_mechanical_midprice_pipeline.py tests/lob/test_candidate_episodes.py -q`: GREEN.
4. Identità verificata sia su ogni componente osservabile sia ad episodio; cancellazioni fisiche deduplicate e totali branch non sommati.

### Task 6 — Writer, report e smoke

**File:** runner e `test_mechanical_midprice_cli.py`.

1. RED per bundle incompleto, leakage di percorsi/credenziali in output distributivo, claim causali e sovrascrittura baseline.
2. Implementare artifact graph, manifest e report con disponibilità/stabilità dei segni; nessun nuovo strict flag.
3. `python -m pytest tests/lob/test_mechanical_midprice_cli.py -q`: GREEN.
4. Smoke da raw sintetico deterministico attraverso il replay reale: confrontare con risultati analitici, non con snapshot inventati dal writer.

### Task 7 — Review e pilot reale autorizzato

1. Review scientifica: set fisso, segni, clock, identità additive, censura e interpretazione del residuo.
2. Review regressioni: equivalenza baseline, lifecycle, schemi vuoti, stabilità degli ID e costo memoria.
3. Eseguire suite completa e diff check; correggere findings concreti e ripetere il gate finale.
4. Solo dopo autorizzazione: pilot reale contenente episodi strict/non-strict, passive/aggressive/mixed, candidati al touch e non al touch, con copertura completa/incompleta. Campione deterministico prima di vedere i risultati.
5. Confrontare baseline e vista osservata, verificare identità e motivi di esclusione; poi valutare il costo di estensione ai tre strumenti. Nessuna promozione automatica nel paper.

## 7. Matrice minima di test scientifici

I numeri di esempio sono casi sintetici analitici, non risultati di mercato.

| Caso | Atteso |
|---|---|
| Bid 100, ask 101 | Midprice 100.5, nessun arrotondamento a tick interi |
| Vendita; candidato bid 100, altro bid 98, ask 104; cancellazione del candidato e nessun altro cambiamento | Midprice osservato 102 → 101; residuo 101 → 101; REV osservata 1, residua 0, meccanica 1 |
| Acquisto; bid 100, candidato ask 104, altro ask 106; cancellazione del candidato e nessun altro cambiamento | Midprice osservato 102 → 103; residuo 103 → 103; REV osservata 1 con `d=-1`, residua 0, meccanica 1 |
| Stesso caso vendita, dalla pubblicazione al pre-fill senza altri eventi | FPM post-placement 0; il salto di inserimento resta una diagnostica diversa |
| Vendita; candidato bid 100, altro bid 98; ask degli altri da 104 a 106 | FPM osservato 1, residuo 1, meccanico 0 |
| Candidato e altro ordine al medesimo best | Il best resta dopo l'esclusione se la quantità visibile altrui è positiva |
| Più candidati e snapshot ripetuti allo stesso livello | Nessuna doppia sottrazione; risultato ottenuto dagli ordini unici attivi |
| ORDERID riutilizzato con first-seen diverso | Escluso solo il lifecycle richiesto |
| Candidati oltre il best o esclusione vuota | Differenza meccanica nulla se il touch non cambia |
| Il best residuo è oltre top-N salvato | Recupero dal replay completo o `insufficient_residual_depth`, mai miglior prezzo inventato |
| Lato residuo mancante, NaN/Inf, locked/crossed | Quote/flag auditabili, componenti invalidi null, non zero |
| Fine dataset prima del target REV | Censurato; nessuna reversione osservata tramite carry-forward |
| Indici diversi allo stesso timestamp | Stati pre/post determinati dal feed, non da ordinamento arbitrario del timestamp |
| Un membro dell'episodio senza residuo | Nessuna media parziale spacciata per decomposizione completa |
| Componenti di segno opposto | Ammessi; nessun clipping o quota artificiale nel range 0–1 |

## 8. Output e comandi proposti

Root futura: `outputs/mechanical_midprice/<timestamp>_<diagnostic_version>/`.

- `config.json`, `command.txt`, `versions.txt`, `manifest.json`, `validation.json`, log;
- `excluded_lifecycle_membership.parquet`: set fisso e origine dei link episodio/cluster;
- `state_requests.parquet`: indici/target richiesti, stato effettivamente selezionato e reason;
- `midprice_states.parquet`: quote osservate/residue, spread, `g`, copertura e flag qualità;
- `cluster_fpm_components.parquet`, `cancel_reversion_components.parquet`, `immediate_quote_components.parquet`;
- `episode_price_components.parquet`, `branch_summary.csv`, `coverage_summary.csv`, `report.md`.

Una riga componente non deve essere contata come episodio. Schemi e selected/observed branches presenti anche in bundle vuoti. Hash di raw, config, codice, membership e baseline; dichiarare che l'esclusione è retrospettiva e che la versione del gate è invariata.

Report minimo: distribuzioni dei componenti firmati, copertura e reasons, tabella dei segni osservato/residuo tra gli episodi valutabili, esempi deterministici e confronto immediato versus successivo. La tabella di segni è una sensibilità diagnostica, **non** un conteggio di veri/falsi positivi o di episodi da eliminare.

Comandi **futuri**, dopo implementazione delle API proposte:

```bash
export PATH="/home/danielemdn/miniconda3/envs/main/bin:$PATH"
hash -r
python scripts/compute_mechanical_midprice_diagnostics.py --help
python scripts/compute_mechanical_midprice_diagnostics.py --config configs/mechanical_midprice_diagnostics.json --validate-only
python -m pytest tests/lob/test_mechanical_midprice.py tests/lob/test_mechanical_midprice_pipeline.py tests/lob/test_mechanical_midprice_cli.py tests/lob/test_replay_observation.py -q
git diff --check
pytest
```

Config incompleta = errore prima di scrivere. Il `pytest` finale è un'invocazione autonoma nell'ambiente del progetto, dopo le verifiche accessorie. La selezione tramite PATH usa gli eseguibili dell'ambiente già esistente senza installare o modificare pacchetti; è stata verificata con Python 3.11.14 e pytest 9.0.2. Non affidarsi a un'attivazione Conda non verificata nelle shell automatizzate. Nessuna compilazione LaTeX necessaria se paper e generatori paper restano intatti.

## 9. Accettazione e arresto

- [ ] Esclusione per lifecycle, stesso set a tutti gli estremi; origine del set integralmente auditabile.
- [ ] Replay osservato invariato: nessun messaggio candidato saltato o modificato.
- [ ] Quote complete e selezioni temporali equivalenti alla baseline.
- [ ] FPM resta post-placement; salti immediati non reinterpretati come FPM.
- [ ] Segni, additività e pesi verificati su casi analitici e percorso end-to-end.
- [ ] Quantità, null, disponibilità del lato residuo e profondità oltre top-N corretti.
- [ ] Episodi misti deduplicati; branch e identity level distinti.
- [ ] Manifest e schemi permettono di ricostruire ogni differenza osservato/residuo.
- [ ] Baseline, soglie, detector, ranking e paper invariati salvo refactor dimostrato equivalente.
- [ ] Due review e suite finale superate prima di proporre il run empirico completo.

**Stop:** impossibilità di ricostruire il book completo ai punti selezionati, mismatch osservato-baseline, lifecycle ambiguo, profondità residua non osservata trattata come certa o qualunque cambiamento non previsto delle metriche canoniche. Quantificare i casi non valutabili invece di forzare una spiegazione.

**Criterio di successo scientifico:** distinguere una variazione spiegata dalla presenza delle quote candidate da un movimento che persiste nella vista residua. Nessuno dei due esiti, da solo, dimostra o esclude manipolazione. Qualsiasi successiva modifica del gate richiede un'altra decisione e una nuova validazione, preferibilmente insieme ai controlli del piano 2.
