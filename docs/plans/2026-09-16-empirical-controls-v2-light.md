# Controlli empirici leggeri — Implementation Plan v2

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task **solo dopo autorizzazione all'implementazione**. Assegnare file disgiunti; replay e contabilizzazione del rischio hanno un solo proprietario. Nessun commit automatico.

**Goal:** Misurare l'associazione descrittiva tra esecuzioni proprie osservate e cancellazione della liquidità eleggibile sul lato opposto, confrontandola con tempo non esposto dello stesso attore e con traslazioni temporali prespecificate.

**Architecture:** Riusare le esecuzioni canoniche congelate dopo certificazione degli input; costruire una cache compatta del rischio con un solo passaggio del replay canonico per partizione-giorno. Calcolare esposizioni, conteggi e sensibilità su intervalli indicizzati, senza ricalcolare il detector completo o ricostruire il book per ogni offset. Scrivere buffer e checkpoint su disco prima di aggregare il report.

**Tech Stack:** Ambiente Python del progetto, Polars, NumPy; standard library per indici e scadenze; pytest per le verifiche future, Matplotlib per una figura statica. Nessuna nuova dipendenza, GPU o framework statistico.

---

## 1. Stato, precedenza e autorizzazioni

- **Aggiornato il 2026-09-16 su richiesta dell'utente.** Il disegno ridotto è recepito in questo piano operativo. Questa richiesta autorizza soltanto l'aggiornamento della documentazione: **non implementare e non eseguire test, preflight, benchmark, pilot o analisi empiriche adesso**.
- Questo è il riferimento operativo per `empirical_controls_v2_light`. Sostituisce la sequenza di implementazione del [piano v1, conservato come archivio](2026-09-09-empirical-negative-controls.md).
- Il [report di revisione](2026-09-16-empirical-controls-review.md) conserva motivazioni e rilievi statici. In caso di differenze operative prevale questo piano, che precisa confini, formati e ordine delle attività.
- Codice, test e `configs/spoofing_empirical_controls.json` esistenti appartengono alla v1 parziale: non sono una v2 funzionante. Nessun checkbox di accettazione è soddisfatto dalla sola stesura del piano.
- Il piano sulla [componente meccanica del midprice](2026-09-09-mechanical-midprice-diagnostics.md) resta separato e non è un prerequisito.
- Non modificare raw, gate strict, WMSCI/MCPS, kernel, identità, episodi canonici, paper, dashboard o risultati storici. Non promuovere `LATEST`, effettuare commit/push o rimuovere il lavoro preesistente.

## 2. Decisioni congelate e fuori perimetro

| Decisione | Contratto v2 |
|---|---|
| Versione | `empirical_controls_v2_light`; config nuova, non reinterpretazione della v1 |
| Natura dell'analisi | Descrittiva/esplorativa, non inferenza causale o validazione di spoofing |
| Popolazione | Tutte le posture attribuibili ed eleggibili; nessun filtro su cancellazione, score, prezzo futuro o strict |
| Profondità / età | Top-10; età massima 90 s con semantica puntuale e di durata esplicita (§3) |
| Esposizione | Finestra di 2 s dalla fine dell'esecuzione aggregata canonica, sul lato opposto |
| Endpoint | Cancellazioni fisiche eleggibili per postura-secondo |
| Contrasti | `own_passive_only − no_own_observed` e `own_aggressive_only − no_own_observed`; entrambi pubblicati |
| Sovrapposizione | `own_mixed` riportato separatamente, non assegnato a una modalità |
| Stratificazione | Strumento, partizione, giorno, attore canonico, lato, fascia fissa di 30 minuti |
| Identità | Risultati client e firm-fallback distinti; riferimento riferito all'attribuzione osservata |
| Sensibilità | Offset `[-60, -30, 0, 30, 60]` secondi; zero ricalcolato sul supporto comune |
| Randomizzazione | Nessuna: niente draw, seed analitico, distribuzione nulla, p-value o CI inferenziale |
| Input canonici | Tutte le esecuzioni qualificanti, non solo matched/strict; cache verificata |
| Esecuzione futura | Partizioni-giorni sequenziali, buffer limitati e checkpoint; pilot autorizzato separatamente |

Rimangono fuori dalla prima v2: confronto own/other, wrong-side, matching/McNemar, hazard model, calibrazione di precision/recall/FPR, probabilità di intento, rescoring FPM/REV/WMSCI/strict alle pseudo-ancore, linkage completo rischio–episodi e grandi griglie di parametri. L'eventuale quantità ritirata per secondo è soltanto diagnostica secondaria entro strumento. Non aggiungere estensioni per compensare un risultato nullo.

## 3. Popolazione, clock e rischio

### 3.1 Liquidità eleggibile, non coorte congelata

Una postura attore-lato è a rischio quando ha almeno un ordine attivo, realmente visibile e nel top-10 con età ammessa. Il rischio riguarda il profilo dinamico: nuovi ordini e rientri possono modificarlo. Non chiamarlo probabilità di ritiro della coorte congelata pre-esecuzione del detector.

- Costruire il rischio anche per attori senza esecuzioni. Non usare `candidate_deceptive_orders` o gli episodi strict per definire l'universo.
- Identificare il lifecycle con partizione, data, order ID e indice canonico di prima osservazione; il riuso di order ID non riusa il lifecycle.
- Conservare il clock d'origine fisico anche se l'ordine esce e rientra nel top-10. Una nuova membership non ringiovanisce l'ordine.
- Il candidato canonico ammette l'età esattamente uguale alla soglia (`spoofing_metrics.py`, `_candidate_deceptive_order_rows`: esclusione con `age_seconds > max_deceptive_order_age_seconds`). La v2 deve ammettere tale uguaglianza per l'evento puntuale pre-evento; per le durate la scadenza a 90 s non concede tempo positivo oltre la soglia. Gestire eventi a esatta scadenza senza cancellare prematuramente la loro eleggibilità puntuale. La v1 del rischio usa una convenzione diversa: non ereditarla senza correzione e fixture.
- Durata di postura, non somma delle vite degli ordini. Membership e quantità possono cambiare senza chiudere lo spell se resta almeno un ordine eleggibile.

### 3.2 Clock e copertura

Riutilizzare l'ordine canonico e il contratto effettivo esportato da `panel.py`/`replay_observation.py`: precedenza timestamp, sort stabile, partizioni e politica di quarantena devono essere registrati e coerenti con la baseline.

- Timestamp mancanti/regressivi interrompono la copertura secondo la policy verificata; non riordinare, clampare o cambiare campo timestamp per ottenere durate positive.
- Conservare tratti contigui di copertura valida (`coverage_epoch_id`, campo proposto), distinti dagli spell. La liquidità assente non è un gap del feed.
- Non estendere la copertura di una partizione usando il primo timestamp della successiva. Fine osservazione, quarantena, condizioni di book invalido e confini devono avere reason esplicite.
- Le fasce di 30 minuti si ancorano a `HH:00`/`HH:30` nel clock dichiarato, non al primo evento osservato. Se il fuso non è stabilito, etichettare `source_clock_timezone_unknown`; non chiamare le fasce opening/closing né convertirle arbitrariamente in orari locali. L'eventuale selezione della negoziazione continua richiede provenance verificata, non orari presunti.
- Conservare tutta la popolazione di rischio nell'audit. Il sottoinsieme con storia sufficiente a classificare l'esposizione è definito separatamente (§4), senza convertire storia mancante in assenza di esecuzione.

### 3.3 Cancellazioni e transizioni

Contare cancellazioni esplicite eleggibili nello stato pre-evento, con quantità effettivamente visibile rimossa. La chiave fisica dell'evento include almeno partizione, indice canonico dell'evento e lifecycle; duplicati dello stesso messaggio non moltiplicano il conteggio. Non deduplicare indiscriminatamente per solo lifecycle se la semantica del feed consente più ritiri parziali distinti.

Fill, modifica, age expiry, uscita dal top-10 e fine copertura non sono cancellazioni. Registrare le transizioni separatamente. Se la causa non è distinguibile, usare `unknown_transition`, senza inventare cancel-replace o intento. Un numero di ordini diminuito non prova un ritiro.

## 4. Confronto osservato entro attore

### 4.1 Gruppo, dominio e quattro stati

Definire `g = (instrument, partition_id, event_date, actor_key, posture_side, time_band_id)`; i campi canonici di identità accompagnano la chiave e devono essere consistenti.

Per ogni blocco temporale `B=[a,b)` ottenuto intersecando fascia e tratto di copertura valida, il dominio del confronto osservato è `J_observed=[a+H,b)`, con `H=2 s`. Il margine iniziale garantisce storia completa nello stesso blocco; non dipende da posture o outcome. Se vuoto, registrare `insufficient_clock_coverage`. Il rischio escluso resta nell'audit. Per questa prima versione si accetta il piccolo ritaglio di bordo invece di implementare imputazioni della storia.

Usare tutte le esecuzioni canoniche osservate nello stesso blocco con attore attribuito uguale e lato opposto alla postura. La finestra parte da `cluster_end_ts`, non dal primo child fill. Unire le finestre della stessa modalità e costruire:

- `no_own_observed`: nessuna finestra propria attribuita attiva;
- `own_passive_only`: solo passive;
- `own_aggressive_only`: solo aggressive;
- `own_mixed`: entrambe.

Sono stati disgiunti che partizionano il rischio nel dominio. `no_own_observed` non significa assenza di trading di mercato o assenza certa di esecuzioni proprie non attribuite. Non inventare modalità o quantità eseguita per il riferimento. Le due righe passive/aggressive di un trade e i child fill devono rispettare membership e ruoli canonici, non essere ricontati come ancore arbitrarie.

L'audit di identità quantifica eventi non attribuibili e finestre cross-level ambigue, con copertura temporale separata: client e firm-fallback non sono automaticamente soggetti economici diversi. I quattro stati descrivono soltanto lo schedule attribuito osservato. Senza mappatura affidabile, non formulare un claim sull'assenza reale di esecuzioni proprie.

### 4.2 Tenere separati punti e durate

- Per il tempo, usare unioni/intersezioni di intervalli senza moltiplicare sovrapposizioni.
- Per la cancellazione osservata, richiedere timestamp nella finestra inclusiva al limite superiore e indice canonico strettamente successivo all'ultimo messaggio del cluster. A timestamp uguale all'apertura, l'ordine feed decide.
- La maschera temporale non deve essere ricalcolata dalla presenza di un ritiro sul bordo. Un evento post-fill allo stesso timestamp non rende esposto il tempo precedente.
- Attribuire ciascuna cancellazione a uno stato puntuale una sola volta nel dominio e nel tipo di analisi; durate e conteggi possono richiedere gestioni diverse dei bordi. Il confine fra fasce segue la convenzione semiaperta del dominio.

### 4.3 Endpoint e aggregazione

Per ogni `g,s` emettere `N[g,s]`, `T[g,s]` e `lambda[g,s]=N[g,s]/T[g,s]` in cancellazioni per postura-secondo.

- `T=0,N=0`: intensità null con reason. Nessun epsilon.
- `T=0,N>0`: anomalia di supporto/risoluzione del clock da investigare prima della pubblicazione; non assegnare tempo pregresso o epsilon per ottenere un tasso finito.
- `T>0,N=0`: intensità zero, non missing.
- Per ogni modalità `m`, `Delta[g,m]=lambda[g,m_only]-lambda[g,no_own_observed]`, soltanto con entrambi i tempi positivi.

Tabella primaria al grain `g`. Per la sintesi entro strumento/modalità/livello di identità, fissare `w[g,m]=min(T[g,m_only], T[g,no_own_observed])` e calcolare la media delle differenze pesata con tali tempi. Non sottrarre tassi aggregati su due popolazioni di attori diverse. Non ottimizzare i pesi sui conteggi.

Pubblicare entrambe le modalità, stato misto, numeratori/denominatori, giorni, actor-day e quota senza confronto. Il riferimento condiviso non è additivo tra branch. Attori senza esposizioni restano nell'audit ma non diventano pseudo-controlli entro-attore di altri soggetti. Distribuzioni per actor-day descrivono eterogeneità, non CI; righe e finestre non sono repliche indipendenti.

### 4.4 Composizione e interpretazione

Audit minimo per stato: count/quantità/età degli ordini eleggibili, spread, profondità e attività trailing di 60 s, con copertura finita/missing separata. Usare solo storia già osservata ai confini di valutazione e tie-break canonico; l'età fra cambi di membership si ricava analiticamente. Non proiettare retroattivamente lo stato alla fine dell'intervallo.

Precalcolare quote/trailing una volta, con as-of e rolling stabili. Non duplicare ogni aggiornamento del mercato in una riga per ogni attore; usare intersezioni o integrali cumulativi. La volatilità trailing è opzionale e non blocca la prima consegna se non è disponibile in modo coerente e leggero; dichiararne l'assenza. Non dividere somme finite per tempo con covariata missing.

Questi dati descrivono composizione, non bilanciamento garantito. Una covariata osservata dopo il fill può esserne già influenzata: non rivendicare un aggiustamento pre-trattamento/causale. La numerosità degli ordini influisce sull'intensità per postura-secondo. Anche un risultato positivo può derivare da gestione legittima di quote/inventario.

## 5. Sensibilità temporale deterministica

### 5.1 Schedule e supporto comune

Usare l'insieme fisso `D={-60,-30,0,30,60}` secondi; il confronto a zero di questa sezione è distinto dal confronto osservato completo del §4. Gli offset sono scelte pragmatiche prespecificate, non tempi calibrati o realizzazioni di una distribuzione nulla.

Per ogni blocco `B=[a,b)` del §4 definire `J_shift=[a+H+60 s,b-60 s)`. Se vuoto, emettere `insufficient_shift_coverage`. Questo ritaglio conservativo è uguale per tutti gli offset e dipende soltanto da clock, copertura e offset, non dalla sopravvivenza delle posture. Permette di valutare `E(t-delta)` e la sua storia di durata `H` interamente nello stesso blocco. Pubblicare la perdita di copertura separatamente dal tempo senza rischio.

- Lasciare rischio e cancellazioni ai tempi osservati; traslare solo gli schedule propri.
- Spostare insieme passive e aggressive dello stesso attore-lato, preservando distanze tra ancore, finestre e overlap. Non campionare separatamente child fill o ancore.
- Non attraversare partizioni, giorni, fasce o gap e non fare wrapping.
- Non imporre corrispondenza esatta a un inizio di intervallo o presenza della postura alla pseudo-ancora. Intersecare `E(t-delta)` con il rischio `R(t)` osservato. La postura può sparire e riapparire: non rifiutare lo schedule per mancata sopravvivenza fra ancore.
- Le pseudo-ancore sono marcatori virtuali: niente sort index, quantità, prezzo, WMSCI, FPM/REV o gate fittizi. Quantità e identità della sorgente possono rimanere provenance, mai nuove esecuzioni.

### 5.2 Convenzione puntuale e aggregazione

Per tutti gli offset, zero incluso, una cancellazione è nella finestra di sensibilità se `anchor+delta < cancel_ts <= anchor+delta+H`. All'apertura virtuale non inventare ordine feed. Contare separatamente le uguaglianze all'apertura non ammesse da quella finestra; se un'altra finestra sovrapposta copre legittimamente il punto, applicare l'unione senza duplicare l'evento. Ogni punto nel dominio riceve comunque uno dei quattro stati secondo questa convenzione, non viene eliminato automaticamente dai conteggi totali.

Per ogni offset ricalcolare i quattro stati, `N`, `T`, intensità e differenze sul medesimo `J_shift`. Non confrontare una traslazione ritagliata con lo zero sul dominio completo.

Per ogni modalità, la sintesi della curva usa soltanto i gruppi con tempi positivi nei due stati a **tutti** gli offset. Fissare su questi gruppi i pesi calcolati allo zero di `J_shift` e riusarli per tutti gli offset. Conservare anche la tabella completa, gruppi esclusi e copertura. Questa selezione di supporto è condizionata al rischio osservato: non dichiararla indipendente dal futuro o generalizzabile all'universo escluso.

Se non esistono gruppi comuni, nessuna curva aggregata: `not_estimable`. Con pochi gruppi o copertura ridotta mostrare la limitazione, senza cercare altri offset. Un picco a zero è evidenza di localizzazione temporale, non di manipolazione; un risultato piatto/nullo resta pubblicabile. Niente p-value, intervalli inferenziali o selezione del ramo/offset più favorevole.

## 6. Architettura e vincoli computazionali

### A. Certificare gli input una volta, poi riusarli

Consumare `execution_metrics.parquet` e `execution_cluster_members.parquet` della baseline congelata, con tutte le esecuzioni qualificanti. Verificare manifest/hash, schema, ruoli esclusivi, identità, quantità, membership, indice finale, clock, partizioni/date e parametri effettivi. Recuperare questi ultimi dalla sezione effettivamente utilizzata dal run, non da default `grid` diversi.

La v2 **non chiama automaticamente** `_recompute_and_verify_canonical_runs` a ogni analisi. La certificazione della baseline è un prerequisito separato e riusabile, identificato da sorgenti/config/codice produttore e verifiche svolte. Se manca evidenza sufficiente, fermarsi e documentare il gate mancante: un hash o un test sintetico non lo sostituiscono. Il futuro audit iniziale deve verificare il contratto degli input consumati; non rieseguire FPM/REV/gate solo per costruire il rapporto conteggio/tempo. Non dichiarare equivalenza integrale con il dirty tree se non dimostrata.

Eventuali modifiche al replay condiviso richiedono regressione/equivalenza sui percorsi canonici interessati. Non produrre un detector v2 alternativo per evitare il confronto.

### B. Replay canonico con osservatore minimo

Riutilizzare `panel.replay_events`, `ReplayHooks`, normalizzazione, `_apply_event` e flush dei residuali. Non duplicare il motore. Evitare nel percorso v2 le copie complete pre/post di `ReplayObservation`: acquisire valori minimi dai hook senza mutare lo stato live né conservarne riferimenti.

Ricostruire comunque tutto il book, inclusi ordini fuori top-10. L'osservatore conserva solo lifecycle/clock e proiezione eleggibile, più i valori pre-evento necessari al ritiro. Aggiornare intervalli attore-lato quando cambiano membership, quantità, eleggibilità o copertura; non a ogni messaggio estraneo. Gestire scadenze con una coda ordinata, aggiornamenti dei livelli e ingressi/uscite senza resettare l'età.

Prima implementazione: proiezione top-N semplice e controllabile. Un indice incrementale per prezzo è ammesso solo dopo profiling se necessario. Non promettere replay lineare: la scansione degli ordini attivi a ogni evento può restare costosa anche senza copie complete.

### C. Chunk, indici e intersezioni

- Filtrare colonne/date con lettura lazy; non leggere tutto e filtrare dopo.
- Partizioni-giorni sequenziali, con inizializzazione completa o checkpoint canonico verificato. Mai avviare un book vuoto a metà giornata.
- `replay_events` rinumera gli indici a ogni invocazione: conservare una mappa esplicita locale→indice originale e usarla per il collegamento ai cluster congelati. Non confondere indici zero-based delle quote con sort index del replay.
- Buffer con limite di righe/byte, flush a file e rilascio della memoria. Non conservare tutti i chunk per una concat finale.
- Una volta persistiti intervalli/membership/cancellazioni, calcolare tutti gli offset senza altri replay.
- Per gruppo ordinare intervalli e finestre, usare unioni, sweep-line o ricerche binarie e integrali cumulativi. Nessun prodotto cartesiano globale attori×messaggi×offset.
- Gli sweep delle intersezioni possono essere proporzionali a intervalli+finestre+cancellazioni dopo l'ordinamento; questa stima non include il costo del book, delle covariate o della lettura.
- Checkpoint autocontenuti, atomicamente rinominati, con schema/hash e chiave di cache comprensiva di input, clock, eligibility e codice rilevante. Cambiare solo gli offset non invalida la cache del rischio, ma invalida gli aggregati. Non riusare output v1 come output v2.

## 7. Superficie dei file e artefatti

### 7.1 File da modificare in futuro

| Percorso | Responsabilità |
|---|---|
| `configs/spoofing_empirical_controls_v2_light.json` — **nuovo, proposto** | Contratto v2, fonti certificate, date, clock, fasce, offset, limiti espliciti di risorse/output |
| `scripts/build_spoofing_empirical_negative_controls.py` | Dispatch esplicito per versione/config, preflight, certificazione, chunk/checkpoint e aggregazione; niente avvio implicito v1 |
| `src/spoofing_detection/lob/withdrawal_risk.py` | Osservatore compatto v2, lifecycle, endpoint puntuali e durate separati, accounting |
| `src/spoofing_detection/lob/negative_controls.py` | Contrasti v2, dominio comune e offset deterministici; preservare helper legacy senza reinterpretazione |
| `src/spoofing_detection/lob/panel.py` | Solo eventuale modifica minima per indici/hook, se indispensabile e verificata; nessuna modifica delle regole LOB |
| `src/spoofing_detection/lob/replay_observation.py` | Percorso v1/riferimento da preservare; non imporne le copie complete al consumer v2 |
| `scripts/build_spoofing_negative_control_report.py` | Loader versionato e report v2 con supporto, unità e limiti; legacy ancora `descriptor_only` dove appropriato |
| `tests/lob/test_withdrawal_risk.py`, `test_empirical_negative_controls.py`, `test_empirical_negative_controls_cli.py`, `test_replay_observation.py`, `test_spoofing_negative_control_report.py` | Fixture e regressioni della nuova semantica/versione |

I nomi di config, campi e artefatti nuovi sono specifiche, non API già implementate. Il JSON v1 non va alterato durante l'aggiornamento del piano. Durante l'implementazione, v1 e v2 devono essere distinguibili e i parametri Monte Carlo v1 rifiutati come opzioni v2, non ignorati silenziosamente.

### 7.2 Bundle v2 proposto

Root nuova: `outputs/spoofing_empirical_controls/<timestamp>_empirical_controls_v2_light/`.

- `manifest.json`, `config.json`, `versions.txt`, `command.txt`, `validation.json`, log e riferimenti alla certificazione baseline;
- `partitions/<partition_key>/`: `risk_intervals.parquet`, `risk_membership_changes.parquet`, `withdrawal_events.parquet`, `transitions.parquet`, `coverage_epochs.parquet`, audit clock/identità e manifest del checkpoint;
- `state_statistics.parquet`: grain `g,analysis_kind,offset_seconds,state` con `N,T,lambda` e null reason;
- `contrast_statistics.parquet`: grain `g,analysis_kind,offset_seconds,comparison_mode` con differenza e supporto;
- `summary.csv`, `coverage.csv`, `balance.csv`, `shift_support.csv`, `report.md`, una figura/tabella statica della sensibilità.

`analysis_kind` distingue `observed_full` e `timing_shift_common_support`; lo zero della seconda non sostituisce il primo. Non usare una colonna `execution_anchor_mode` per attribuire un fill al riferimento: usare `comparison_mode` per il contrasto. Conservare sorgenti e chiavi canoniche per possibili audit futuri senza materializzare il linkage completo agli episodi.

Schema e branch configurate devono esistere anche con zero righe. L'audit distingue rischio totale, esclusioni per clock/storia/bordi, dominio osservato, dominio comune e gruppi non confrontabili. Nessuna tabella per mille schedule simulati. Il report finale rilegge e riconcilia i file persistiti, non si fida soltanto dei contatori in memoria. Non pubblicare il bundle finale prima di hash, accounting e loader validi; un checkpoint parziale non è un risultato completo.

## 8. Attività di implementazione — tutte future

Per ogni attività: scrivere fixture/asserzioni → eseguire e documentare il fallimento comportamentale atteso → modifica minima → ripetere il test → diff. Nessuna attività seguente è autorizzata dalla richiesta di aggiornamento del piano. Non incollare nei documenti risultati di test non eseguiti.

### Task 1 — Versione, input certificati e limiti

**File:** config v2, runner, `test_empirical_negative_controls_cli.py`.

1. Aggiungere fixture per v1/v2 incompatibili, certificazione assente, schema/hash/ruoli/date errati, input vuoto, output già esistente e budget mancante.
2. Implementare preflight senza replay o scrittura; certificazione riusabile e dipendenze precise.
3. Rendere espliciti selezione config/versione, cap di buffer, limite RAM e timeout prima di un futuro run reale. Non impostare cap permissivi presunti dal vecchio pilot.
4. Verificare che il percorso v2 non invochi il rescoring completo del detector e che input invalidi falliscano prima di qualsiasi output.

**Comando futuro:** `python -m pytest tests/lob/test_empirical_negative_controls_cli.py -q`.

### Task 2 — Rischio compatto ed equivalenza del replay

**File:** `withdrawal_risk.py`, eventuale hunk minimo `panel.py`, test rischio/replay.

1. Fixture con attore senza fill, ordini fuori top-N che rientrano, età esatta 90 s, riuso ID, stop inattivo, residuali, clock regressivo e cambio partizione.
2. Implementare osservatore sui hook canonici, proiezione minima, intervalli per cambiamento e scadenze senza snapshot completi.
3. Confrontare con una contabilità di riferimento su fixture raw: stessi lifecycle, cancellazioni, tempi e copertura dopo le correzioni semantiche dichiarate. Non imporre identità del numero di righe con la rappresentazione v1 espansa.
4. Forzare buffer minuscoli: chunking e indici locali/originali non devono cambiare risultati o output canonici del detector.

**Comando futuro:** `python -m pytest tests/lob/test_withdrawal_risk.py tests/lob/test_replay_observation.py tests/lob/test_reconstruction.py tests/lob/test_spoofing_metrics.py -q`.

### Task 3 — Stati osservati e differenze entro attore

**File:** `withdrawal_risk.py`, `negative_controls.py`, relativi test.

1. Fixture con finestre passive/aggressive sovrapposte, candidati dopo l'esecuzione, ritiro senza fill, identità ambigue e attore senza tempo esposto.
2. Riprodurre il caso fill/cancel allo stesso timestamp: correggere separatamente classificazione puntuale e durata. A rischio e schedule fissati, cambiare l'attribuzione dei punti non modifica `T`.
3. Implementare unioni, quattro stati, dominio osservato, `N/T`, differenze e pesi del §4; null/zero distinti.
4. Verificare somme di tempi e cancellazioni, assenza di duplicazioni, strati separati e nessuna equivalenza abusiva fra nuova domanda e contrasti v1.

**Comando futuro:** `python -m pytest tests/lob/test_withdrawal_risk.py tests/lob/test_empirical_negative_controls.py -q`.

### Task 4 — Offset fissi e supporto comune

**File:** `negative_controls.py`, `test_empirical_negative_controls.py`.

1. Fixture con clock irregolare, gap, bordo fascia, spell che chiude fra ancore e finestra che resta censurata: niente snapping e niente selezione sulla sopravvivenza.
2. Implementare `J_shift`, traslazione congiunta delle modalità, tie convention comune e ricalcolo a zero.
3. Verificare i risultati contro un conteggio/intersezione semplice di riferimento su fixture; un disallineamento temporale deve poter cambiare numeratori e denominatori senza cambiare il book.
4. Verificare supporto comune, pesi fissi a zero, insiemi vuoti e impossibilità di sostituire uno schedule unsupported con un negativo. Ripetizioni deterministiche devono coincidere senza RNG.

**Comando futuro:** `python -m pytest tests/lob/test_empirical_negative_controls.py tests/lob/test_negative_controls.py -q`.

### Task 5 — Balance, checkpoint e bundle completo

**File:** runner, report, test CLI/report.

1. Aggiungere fixture raw end-to-end, zero righe e checkpoint interrotto/corrotto; non fabbricare metriche detector di comodo per provare il percorso completo.
2. Implementare balance indicizzato, flush/ripresa, manifest, schemi vuoti, output per versione e report.
3. Rileggere i file: riconciliare popolazione, domini, tempi, quattro stati, conteggi, contrasti e supporto per ogni offset; un manifest coerente con un solo CSV non basta.
4. Verificare report senza claim di intento, causalità, FPR, significatività o validazione del detector. Nessun output v1 pubblicato come v2.

**Comando futuro:** `python -m pytest tests/lob/test_empirical_negative_controls_cli.py tests/lob/test_spoofing_negative_control_report.py -q`.

### Task 6 — Review e verifiche software

1. Review scientifica: selezione, clock, supporto, identità, dipendenza, endpoint dinamico e interpretazione.
2. Review codice/risorse: memoria, complessità, indici, checkpoint, compatibilità e regressioni.
3. Chiudere i rilievi concreti, ripetere i test focalizzati, poi suite completa e controllo del diff. La revisione statica del report non vale come approvazione dell'implementazione v2.

**Comandi futuri**, nell'ambiente di progetto già verificato all'avvio del lavoro: `python -m pytest` e, separatamente, `git diff --check`. Non installare/aggiornare dipendenze solo per eseguire il piano.

### Task 7 — Pilot e scala, con autorizzazioni distinte

1. Prima di qualsiasi dato reale: autorizzazione esplicita al pilot, gate input e test chiusi, budget RAM/tempo e selezione della partizione fissati senza outcome.
2. Misurare wall time, picco RSS, swap/pressione, dimensioni dei buffer, righe per evento, intervalli compatti, copertura e supporto. Fermarsi al budget senza alzare progressivamente RAM/swap o esentare il processo dalle protezioni del desktop.
3. Verificare gli output reali persistiti e stimare il costo residuo senza assumere linearità dal solo numero di righe raw. Se non sostenibile, correggere il collo di bottiglia misurato, non ampliare il run.
4. Solo dopo approvazione del pilot e nuova autorizzazione alla scala: completare sequenzialmente date/strumenti congelati. Un sottocampione cambia la popolazione e va dichiarato; nessuna scelta dei giorni sulla base della separazione ottenuta.

Non includere un comando di run reale pronto da eseguire prima che esistano config v2, budget e certificazione. Nessuna promozione automatica degli output nel paper.

## 9. Accettazione e condizioni di arresto

- [ ] Versione/config/certificazione input distinguibili dalla v1 e validate prima del replay.
- [ ] Popolazione a rischio senza selezione su outcome, inclusi attori senza esecuzioni.
- [ ] Lifecycle, età al confine, ordine feed, copertura e indici originali corretti.
- [ ] Tempi e punti separati; quattro stati disgiunti e accounting riconciliato.
- [ ] Zero/null/supporto insufficiente espliciti, nessuna durata o esecuzione inventata.
- [ ] Passive/aggressive e identità separate; entrambi i contrasti riportati.
- [ ] Offset fissi, dominio comune, zero ricalcolato, nessuna selezione sulla sopravvivenza fra ancore.
- [ ] Cache del rischio riusabile senza nuovi replay per offset/report; chunking e checkpoint verificati.
- [ ] Balance/missing/censura/ambiguità e quote di popolazione escluse leggibili.
- [ ] Report e file persistiti coerenti; nessun p-value/CI o claim di validazione non supportato.
- [ ] Due review, test end-to-end e suite finale completati prima del pilot.
- [ ] Pilot reale autorizzato, misurato e sostenibile prima della scala.

**Stop:** provenienza o ruoli incompatibili, certificazione input insufficiente, indici non riconciliati, impossibile inizializzazione, clock non auditabile, denominatori dipendenti dagli outcome, quantità/tempi inventati, modifica involontaria del detector o budget di risorse superato. Un dominio/gruppo privo di supporto è invece un esito `not_estimable` da riportare, non un motivo per cambiare a posteriori offset o popolazione.

**Consegna scientifica minima:** tabella delle intensità e differenze entro attore, figura/tabella della sensibilità temporale e audit della copertura. Un'associazione nulla o contraria all'atteso soddisfa il piano tanto quanto un'associazione positiva, purché accounting e limiti siano corretti.
