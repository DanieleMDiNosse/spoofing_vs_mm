# Controlli empirici end-to-end — ARCHIVIO Implementation Plan v1

> **ARCHIVIO — sostituito il 2026-09-16:** il [piano operativo v2](2026-09-16-empirical-controls-v2-light.md) recepisce il disegno ridotto richiesto dall'utente e sostituisce tutte le attività e i criteri operativi qui sotto. Questo file conserva soltanto la v1 storica: non implementarne automaticamente task, Monte Carlo o rescoring completo. Il [report di revisione](2026-09-16-empirical-controls-review.md) documenta le motivazioni. L'aggiornamento riguarda solo il piano; codice/config restano v1 e implementazione, test e run reali richiedono autorizzazione distinta.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Assegnare file disgiunti; la parte di replay condivisa ha un solo proprietario. Non implementare prima di autorizzazione esplicita.

**Goal:** Verificare se il ritiro della liquidità candidata è temporalmente associato alle esecuzioni dell'attore focale oltre a quanto osservato durante attività di mercato comparabile, senza trattare i controlli come casi legittimi certificati.

**Architecture:** Affiancare al detector un pannello del tempo a rischio, costruito dal replay canonico anche in assenza di esecuzioni proprie. Calcolare esposizioni reali, ritiri osservati e statistiche placebo end-to-end in un runner isolato. Conservare gli episodi del detector come unità primaria dei risultati del detector e collegarli, senza confonderli, alle nuove unità di rischio.

**Tech Stack:** Python 3.11 dell'ambiente di progetto, Polars, NumPy, SciPy e pytest; Matplotlib solo per il report. Nessuna nuova dipendenza nella v1.

---

## 1. Stato, perimetro e decisioni

- **Stato storico v1:** archiviato, non più piano operativo. Per decisioni, attività e autorizzazioni usare la v2 collegata sopra; le sezioni successive sono conservate come riferimento storico.
- Questo è il **punto 2** della discussione sull'attendibilità. Il piano separato [componente meccanica del midprice](2026-09-09-mechanical-midprice-diagnostics.md) non è un prerequisito della v1.
- Nessuna modifica al gate strict, a WMSCI/MCPS, alle soglie, ai kernel, alle identità, agli episodi canonici o al paper.
- Nessuna classificazione automatica di intento, probabilità di spoofing, precision/recall o false-positive rate. Le finestre di controllo possono contenere condotte sospette non etichettate.
- Preservare i due rami `passive` e `aggressive`, la distinzione client/firm e i confini strumento/partizione/giorno. Se esiste un identificativo verificato di sessione, usarlo; non chiamare una partizione giornaliera una sessione senza evidenza.
- Non reintrodurre come analisi principale un matching 1:1 seguito da McNemar: molti abbinamenti possono dipendere dagli stessi attori/giorni e i controlli possono essere scarsi. Il tempo a rischio completo è il disegno principale; gli abbinamenti sono diagnostici secondari.
- La v1 è **descrittiva/esplorativa**, con audit della comparabilità. Modelli di hazard, inferenza confermatoria e calibrazione del classificatore sono esclusi dalla v1.
- Non eseguire commit, push, promozione di `LATEST` o rigenerazione di artefatti storici senza richiesta.

## 2. Evidenza dal codice attuale

Riferimenti verificati durante la pianificazione; le righe vanno ricontrollate prima di implementare.

| Fonte | Comportamento corrente | Conseguenza |
|---|---|---|
| `src/spoofing_detection/lob/negative_controls.py:6–29` | Aggiunge un indice traslato o un lato alternativo | Non ricalcola outcome o statistiche |
| `scripts/build_spoofing_negative_control_report.py:18–34` | Conta descrittori e rinvia lo scoring | Non è una validazione empirica già eseguita |
| `tests/lob/test_negative_controls.py:8–21` | Verifica offset e inversione del lato | Servono test del percorso completo |
| `src/spoofing_detection/lob/spoofing_metrics.py:989–1198` | Replay, estrazione candidati ai fill, cancellazioni e cluster | Riutilizzare la semantica; il solo `candidate_df` è selezionato sulle esecuzioni |
| `src/spoofing_detection/lob/candidate_episodes.py:157–209` | Collega cluster tramite lifecycle e assegna cancellazioni uniche | Gli episodi ex-post non definiscono la popolazione no-fill |
| `src/spoofing_detection/lob/spoofing_metrics.py:1784–1876` | Reversione cancel-anchored con copertura e pesi quantità/ritardo | Riutilizzare i controlli temporali, non fabbricare esecuzioni |
| `requirements_simulations.txt` | Stack scientifico già dichiarato | Nessun nuovo framework necessario |

Il template `configs/spoofing_detection_parameters.json` contiene sezioni `metrics` e `grid` con valori diversi. Recuperare parametri effettivi e input dal run selezionato, non dai default di un'altra sezione.

## 3. Contratto scientifico

### 3.1 Domanda ed endpoint

**Domanda primaria:** quando una postura eleggibile è viva, il tasso osservato di cancellazione cambia dopo un'esecuzione propria rispetto a esecuzioni di altri attori e a tempo senza esecuzioni qualificanti?

Endpoint v1 principale, per strato e stato di esposizione:

`withdrawal_intensity = numero di cancellazioni fisiche eleggibili / secondi di postura a rischio`.

- È un'intensità descrittiva di eventi, non una probabilità né un hazard individuale stimato.
- Il numeratore conta ogni cancellazione fisica una sola volta nel suo strato. Il denominatore è tempo di postura attore-lato, non somma delle vite di tutti gli ordini del profilo.
- La quantità visibile ritirata per secondo a rischio è una diagnostica secondaria; etichettare le unità e non confrontare quantità grezze tra strumenti.
- Se il tempo a rischio è nullo, intensità `null`, non zero e non epsilon. Un numeratore positivo con denominatore nullo è un errore da investigare.
- Frequenze di ritiro entro l'orizzonte dopo un'ancora e frazioni di quantità rimossa sono secondarie; riportare censura, eventi concorrenti e denominatori. Non sommare finestre sovrapposte come osservazioni indipendenti.
- Differenze tra esposizioni restano associazioni: composizione delle posture, market making, inventario non osservato e shock comuni possono spiegarle.

### 3.2 Unità di rischio costruita senza outcome

Nuovi identificatori **proposti**, distinti da `episode_id`:

1. `risk_spell_id`: tratto continuo entro attore, lato postura, partizione e giorno nel quale esiste almeno un ordine eleggibile. Inizia quando il profilo diventa eleggibile; termina quando non resta alcun ordine eleggibile, finisce la copertura o si incontra un confine.
2. `risk_interval_id`: segmento di durata positiva in cui appartenenza del profilo ed esposizione sono costanti. Segmentare anche alle scadenze di età/finestra che cadono tra due messaggi, non solo alle righe del feed.
3. `risk_membership`: lifecycle eleggibili di ciascun intervallo; chiave almeno `(partition_id, event_date, order_id, first_seen_sort_index)`, con identità canonica e lato.

Elegibilità: ordini effettivamente visibili, attivi, nel top-N osservato e con età compatibile con la configurazione. Queste condizioni si verificano sullo stato precedente all'evento; non richiedono ritiro, FPM, reversione o passaggio del gate futuro.

Le modifiche di membership segmentano l'intervallo ma non moltiplicano tempo o cancellazioni. Un ritiro parziale può lasciare vivo lo spell. Nuovi lifecycle e rientri nel top-N sono espliciti; i nuovi ordini non entrano retroattivamente nella coorte congelata di una finestra secondaria.

`episode_id` viene collegato **dopo** la costruzione del rischio attraverso lifecycle e indici: uno spell può non avere alcun episodio del detector. L'assenza di `episode_id` è valida per no-fill.

### 3.3 Clock ed esposizioni

- L'ordine causale è quello canonico del feed, con `sort_index` e partizione. Timestamp uguali non sono intercambiabili.
- Verificare monotonicità del clock dentro la partizione prima di costruire durate. Per regressioni temporali non risolte, segmentare/quarantinare ed emettere una ragione; non riordinare arbitrariamente i messaggi per rendere positive le durate.
- Una esecuzione qualificante usa ruolo, lato e quantità reali. La finestra di esposizione si apre alla **fine del cluster**, non al primo child fill, e termina dopo `withdrawal_window_seconds`.
- Un ritiro è successivo solo se soddisfa contemporaneamente `cancel_ts >= cluster_end_ts`, `cancel_sort_index > cluster_last_sort_index` e il limite superiore di clock. Conservare la convenzione inclusiva del limite superiore per l'evento puntuale.
- Distinguere `own_passive`, `own_aggressive`, `other_passive`, `other_aggressive`, assenza di esposizione e combinazioni sovrapposte. Conservare la maschera completa degli stati; non assegnare arbitrariamente una branch a una sovrapposizione.
- Contrasti semplici: own-only versus other-only versus no-qualifying-fill, con composizioni miste/sovrapposte riportate a parte. La popolazione esclusa dai contrasti deve restare quantificata.
- `other` richiede un'identità osservata distinguibile da quella focale **alla stessa granularità**. Non dedurre che una firm fallback e un client siano soggetti diversi solo perché hanno namespace diversi; i casi non identificabili sono `identity_ambiguous`.
- Le due righe dello stesso trade e sweep multi-fill non devono moltiplicare l'esposizione. Usare trade ID, membership e ruolo verificati, senza cambiare la costruzione canonica dei cluster.
- I tempi no-fill hanno `execution_anchor_mode = null` e quantità eseguita `null`. Nei confronti branch-specific la branch appartiene al **contrasto**, non a un'esecuzione inesistente. Il medesimo riferimento no-fill eventualmente riusato va dichiarato non additivo.

### 3.4 Outcome, competizione e osservabilità

Osservare cancellazioni esplicite eleggibili al loro stato pre-evento, con chiave fisica canonica e quantità visibile rimossa. Non derivare il ritiro dalla sola diminuzione del numero di ordini: potrebbe essere fill, modifica o uscita dai livelli osservati.

- Fill degli ordini candidati, modifica, scadenza, perdita di eleggibilità e fine copertura sono transizioni/eventi concorrenti registrati separatamente; non sono cancellazioni per definizione.
- Se i codici non distinguono una causa, usare `unknown_transition`; nessun collegamento cancel-replace o intento inventato.
- Zero ritiro richiede osservazione del periodo a rischio. Fine dataset, lato assente, book invalido e clock ambiguo non sono negativi.
- Per finestre ancorate secondarie, censurare al primo evento che impedisce di osservare il target secondo il protocollo; non richiedere in selezione che la postura sopravviva fino a fine finestra. Conservare sia durata osservata sia reason.
- Non attribuire tutti gli altri ritiri al winning cluster del detector: il nuovo endpoint include ritiri anche senza esecuzione. L'attribuzione canonica serve al linkage, non a definire l'outcome dei controlli.

### 3.5 Confronti e placebo non circolari

**Popolazione osservata:** tutte le posture eleggibili e tutte le esecuzioni qualificanti, inclusi casi non strict. Mai scegliere i casi perché hanno già ritiro o percorso di prezzo favorevole.

**Comparabilità:** prima del confronto esporre spread in tick, profondità, quantità/età/count del profilo, fascia oraria, attività e volatilità calcolate solo da passato osservato. Congelare strata/caliper in una configurazione prima di esaminare gli outcome. Se mancano supporto comune o calibrazione, presentare risultati stratificati non aggiustati, con avvertenza, invece di dichiarare i controlli comparabili.

**Placebo temporale v1:** mantenere invariati book, ritiri e identità; spostare/campionare solo lo schedule delle esposizioni proprie entro blocchi dello stesso attore-giorno-lato e fascia di attività prespecificata. Preservare distanze tra eventi del blocco, durata e branch delle finestre; non permutare child fill isolati. Non attraversare chiusure, gap o confini e non fare wrapping circolare della giornata.

- Mappare ogni pseudo-ancora a un confine osservato secondo una regola deterministica dichiarata. Rifiutare schedule non ordinati o senza supporto at-risk al momento dell'ancora, non quelli che hanno un esito indesiderato.
- Non richiedere sopravvivenza fino a `H`: censurare dopo l'ancora in modo uguale ai dati reali. Gli indici, lo stato e le covariate scelti a una data ancora non possono usare il suo futuro.
- Un test di indipendenza dal futuro vale a **ancora e storia pre-ancora fissate**: cambiare ritiri futuri non deve cambiare selezione/membership pre-ancora. Il pannello a rischio successivo può invece cambiare legittimamente.
- Ricostruire per ogni draw esposizioni, intersezioni con intervalli a rischio, numeratori, denominatori e statistica aggregata. Nei descrittori `placebo_draw_id` non ci sono nuove esecuzioni economiche.
- Una distribuzione ottenuta così è un riferimento esplorativo condizionato al book osservato, non un test esatto senza una giustificazione di scambiabilità. Riportare draw rifiutati, motivi, differenze di supporto e copertura.
- `wrong_side` è solo una falsificazione secondaria: richiede il profilo realmente osservato sull'altro lato e metriche ricalcolate. Non usarlo per dimostrare selettività se il gate lo respinge per costruzione.

**Confine del rescoring completo:** rieseguire il percorso canonico su tutte le esecuzioni **reali** per validare membership, quantità, cancellazioni, FPM/REV e gate, senza selezionare solo i positivi. Per no-fill e pseudo-ancore ricalcolare solo endpoint definiti: nessun WMSCI, rapporto su quantità eseguita o strict gate con fill fittizio. Non confrontare questi endpoint come se fossero lo stesso score.

## 4. Disegno statistico e decisioni prima del run

- Rendere obbligatori `design_version`, fonti/hash, clock, orizzonti, eligibility policy, strata, seed, numero draw e selezione dei giorni. I parametri detector sono quelli del run congelato.
- Proposta riproducibile per il solo generatore casuale: `seed = 20260909`; usare `numpy.random.default_rng` locale e substream deterministici, senza reseed dentro i loop.
- Il numero di draw è un parametro computazionale esplicito; smoke e run empirico devono avere etichette diverse. Non scegliere il numero in base al risultato.
- Report obbligatorio: strumenti, giorni distinti, attori, actor-day, spell, secondi a rischio, eventi e censura; nessun conteggio di righe come numerosità indipendente.
- Prima analisi: intensità e differenze assolute stratificate, distribuzioni placebo e diagnostiche pre/post. Un pre-trend va valutato sul rischio realmente esistente prima del fill, non soltanto sui profili sopravvissuti fino al fill.
- Intervalli di variabilità tra draw non sono confidence interval sul parametro di popolazione. Con un solo giorno non produrre inferenza day-clustered confermatoria.
- **Estensione successiva, non v1:** modello di hazard con aggiustamento pre-evento, competizione esplicita, split cronologico e incertezza per giorno/blocco, eventualmente multiway actor/day. Richiede un piano di identificazione e dati sufficienti; non si attiva automaticamente.

## 5. File previsti e confini di proprietà

Tutti i nomi nuovi sotto sono **proposte**, non API già esistenti.

| Azione | Percorso | Responsabilità |
|---|---|---|
| Creare | `src/spoofing_detection/lob/withdrawal_risk.py` | Spell, membership, intervalli, esposizioni e outcome |
| Estendere | `src/spoofing_detection/lob/negative_controls.py` | Schedule placebo e aggregazione; preservare compatibilità dei vecchi helper descrittori |
| Creare | `src/spoofing_detection/lob/replay_observation.py` | Minima orchestrazione osservabile estratta dal replay metriche, solo se necessaria; condivisa col piano 3 |
| Modificare localmente | `src/spoofing_detection/lob/spoofing_metrics.py` | Collegamento all'orchestrazione condivisa con default equivalente e test differenziali |
| Creare | `scripts/build_spoofing_empirical_negative_controls.py` | Preflight, replay, calcolo e scrittura isolated-run |
| Estendere | `scripts/build_spoofing_negative_control_report.py` | Lettura del bundle validato; distinguere esplicitamente report legacy di descrittori |
| Creare | `configs/spoofing_empirical_controls.json` | Disegno separato, mai override silenzioso della baseline |
| Creare | `tests/lob/test_withdrawal_risk.py` | Semantica time-at-risk |
| Creare | `tests/lob/test_empirical_negative_controls.py` | Placebo e aggregazioni |
| Creare | `tests/lob/test_empirical_negative_controls_cli.py` | Preflight e percorso end-to-end |
| Creare/condividere | `tests/lob/test_replay_observation.py` | Equivalenza del replay e callback non mutanti |
| Estendere | `tests/lob/test_negative_controls.py`, `tests/lob/test_spoofing_negative_control_report.py` | Compatibilità e report senza claim indebiti |

Nessun secondo motore LOB: riusare `sort_events`, `normalize_event`, `_apply_event`, gestione dei residuali e ordine di flush. Non leggere solo gli attori selezionati dalle esecuzioni quando si costruisce il rischio: includere anche attori con postura eleggibile e nessun fill. Non conservare riferimenti mutabili al dizionario `active_orders` tra eventi.

## 6. Task di implementazione e verifica

Ogni task procede per passi piccoli: test RED → verifica del fallimento atteso → modifica minima → test GREEN → diff. Un refactor condiviso deve chiudere il gate di equivalenza prima di iniziare la statistica. Nessun commit automatico.

### Task 1 — Contratto e preflight

**File:** config nuova, runner nuovo, `test_empirical_negative_controls_cli.py`.

1. Scrivere test per metadati mancanti/incompatibili, input vuoto schema-valido, split con confini condivisi, clock ambiguo e output già esistente.
2. Eseguire `python -m pytest tests/lob/test_empirical_negative_controls_cli.py -q`: RED per validazione non implementata.
3. Implementare parsing esplicito, schema e `--validate-only` senza creazione di output.
4. Ripetere il comando: GREEN; input invalido rifiutato prima del replay o delle scritture.

### Task 2 — Replay osservabile equivalente

**File:** `replay_observation.py`, integrazione minima in `spoofing_metrics.py`, `test_replay_observation.py`.

1. Fissare fixture realiste con passive/aggressive, stesso timestamp, cambio partizione, stop inattivo e residuali aggressivi.
2. RED per osservazione pre/post ai confini canonici e per un attore senza esecuzioni.
3. Estrarre solo l'orchestrazione indispensabile; nessuna modifica alle regole `_apply_event`.
4. `python -m pytest tests/lob/test_replay_observation.py tests/lob/test_reconstruction.py tests/lob/test_spoofing_metrics.py -q`: GREEN e uguaglianza di schema/valori degli output canonici senza osservatore e con osservatore non mutante.

### Task 3 — Rischio e transizioni

**File:** `withdrawal_risk.py`, `test_withdrawal_risk.py`.

1. RED per ingresso/uscita top-N, scadenza età tra messaggi, cancellazione parziale, fill, modifica, riuso ORDERID e fine giornata.
2. Implementare spell e intervalli con membership e reason; endpoint esclusivamente da cancellazioni osservate.
3. `python -m pytest tests/lob/test_withdrawal_risk.py -q`: GREEN.
4. Verificare partizione disgiunta del tempo, somme delle durate, chiavi fisiche uniche e null in mancanza di copertura.

### Task 4 — Esposizioni osservate e contrasti

**File:** `withdrawal_risk.py`, `test_withdrawal_risk.py`.

1. RED per finestra dalla fine cluster, cancel precedente allo stesso timestamp, own/other ambiguo, trade a due righe e mixed branches.
2. Implementare maschere e intersezioni temporali senza duplicazione del tempo; no-fill conserva branch/quantità null.
3. Ripetere i test del Task 3 e aggiungere fixture che cambia il futuro senza cambiare eleggibilità alla stessa ancora.
4. Verificare che una cancellazione possa essere contata anche senza qualsiasi cluster del detector.

### Task 5 — Placebo realmente ricalcolati

**File:** `negative_controls.py`, `test_empirical_negative_controls.py`.

1. RED per seed, confini, no wrapping, preservazione delle sequenze e ricalcolo dei denominatori.
2. Implementare campionamento a blocchi e nuova intersezione col pannello di rischio; annotare rifiuti/supporto.
3. `python -m pytest tests/lob/test_negative_controls.py tests/lob/test_empirical_negative_controls.py -q`: GREEN.
4. Fixture con ritiri allineati alle esecuzioni: traslando le esposizioni cambia la statistica effettiva; una copia dello score reale deve fallire il test. Fixture senza allineamento: il metodo non deve produrre separazione per definizione.

### Task 6 — Artefatti, report e smoke end-to-end

**File:** entrambi i runner/report, test CLI/report.

1. RED per artifact graph incompleto, totali incoerenti e falsa etichetta FPR/precision/intent.
2. Implementare bundle e report con numeratori, denominatori, supporto e reason. Il report legacy rimane marcato `descriptor_only`.
3. `python -m pytest tests/lob/test_empirical_negative_controls_cli.py tests/lob/test_spoofing_negative_control_report.py -q`: GREEN.
4. Eseguire il nuovo runner su fixture raw deterministica attraverso il replay reale, non su metriche fabricate. Contare in codice membership, cancellazioni, tempi e risultati per draw.

### Task 7 — Audit prima di qualsiasi run completo

1. Review scientifica: selezione, clock, censura, composizione degli stati, dipendenza e interpretazione.
2. Review codice: equivalenza baseline, null, identità, input vuoti, determinismo, sicurezza e uso memoria.
3. Correggere solo findings concreti, rieseguire test focalizzati e suite completa.
4. Solo dopo nuova autorizzazione: pilot reale circoscritto con inizializzazione corretta del book; poi run completo in root nuova. Nessuna calibrazione delle soglie usando gli esiti del pilot.

## 7. Bundle atteso e comandi proposti

Root futura: `outputs/spoofing_empirical_controls/<timestamp>_<design_version>/`.

- `config.json`, `versions.txt`, `command.txt`, `manifest.json`, log con exit status;
- `risk_spells.parquet`, `risk_intervals.parquet`, `risk_membership.parquet`;
- `exposure_intervals.parquet`, `withdrawal_events.parquet`, `episode_risk_links.parquet`;
- `placebo_schedules.parquet`, `placebo_statistics.parquet`, `coverage_and_balance.csv`;
- `stratum_statistics.csv`, `report.md`, `validation.json`, eventuali figure statiche.

Schema, configurazione selezionata e branch osservate devono essere emessi anche con zero righe. Il manifest registra snapshot del codice, input raw/kernel/config con hash, seed, clock, policy di rischio, unità di ogni tabella e ragioni delle esclusioni; non contiene credenziali.

Comandi **futuri**, dopo implementazione delle API proposte:

```bash
export PATH="/home/danielemdn/miniconda3/envs/main/bin:$PATH"
hash -r
python scripts/build_spoofing_empirical_negative_controls.py --help
python scripts/build_spoofing_empirical_negative_controls.py --config configs/spoofing_empirical_controls.json --validate-only
python -m pytest tests/lob/test_withdrawal_risk.py tests/lob/test_empirical_negative_controls.py tests/lob/test_empirical_negative_controls_cli.py tests/lob/test_replay_observation.py -q
git diff --check
pytest
```

La configurazione nuova deve specificare i path esatti recuperati dai metadati della baseline scelta; nessun run empirico parte con un template incompleto. Il `pytest` finale va eseguito come invocazione autonoma dopo le altre verifiche, nell'ambiente indicato. La selezione tramite PATH usa gli eseguibili dell'ambiente già esistente senza installare o modificare pacchetti; è stata verificata con Python 3.11.14 e pytest 9.0.2. Evitare di assumere funzionante l'attivazione Conda nelle shell automatizzate: in questa sessione ha incontrato uno stato ereditato incoerente.

## 8. Accettazione e condizioni di arresto

- [ ] La popolazione a rischio contiene anche posture senza esecuzioni e non è filtrata sul gate/outcome.
- [ ] Intervalli disgiunti, durate positive, esposizioni/cancellazioni con ordine feed e confini verificati.
- [ ] Client/firm e rami non confusi; sovrapposizioni e no-fill espliciti.
- [ ] Nessuna quantità eseguita, branch o prezzo di esecuzione inventati.
- [ ] Statistiche placebo ricalcolate con seed e supporto riproducibili, non copiate.
- [ ] Censura/competizione separate dai negativi; nessun riempimento silenzioso dei null.
- [ ] Supporto comune e balance mostrati; nessun claim confermatorio da singolo giorno o righe dipendenti.
- [ ] Replay e output detector invariati; raw, baseline, dashboard e paper non sovrascritti.
- [ ] Due revisioni e test di integrazione/suite finale superati prima di proporre un rerun empirico.

**Stop:** mapping di identità/ruolo non risolvibile, clock incoerente, impossibilità di ricostruire la postura prima dell'ancora, contaminazione del denominatore da outcome o refactor che cambia gli output canonici. Produrre un audit di fattibilità, non un risultato apparentemente validato.

**Criterio di successo scientifico:** ottenere una misura auditabile dell'associazione, anche nulla o contraria alle attese. Un detector che non si separa dai controlli è un risultato da riportare, non un motivo per ritoccare a posteriori il disegno.
