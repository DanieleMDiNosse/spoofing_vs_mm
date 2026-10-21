# Riproduzione del paper

Il manoscritto attivo è `paper/spoofing_simple.tex`, con i sorgenti inclusi in
`paper/sections/`. Le analisi sono esplorative: i candidati non sono etichette di
manipolazione e i confronti di prezzo non identificano effetti causali.

## Distribuzione del codice e dati riservati

Il branch di pubblicazione contiene i sorgenti, non i dati CONSOB né gli output
empirici. `data/`, `outputs/`, `history/`, `astra_metric_audit/`, `reports/`,
`paper/generated/`, `paper/reviews/`, i PDF e i file tabulari sono esclusi
dal `.gitignore` e dallo snapshot pubblicato. Le sezioni seguenti descrivono
l'ambiente locale autorizzato, in cui tali dipendenze sono disponibili.

Un clone del repository pubblico da solo non consente di ricostruire i risultati
empirici, compilare le figure del paper o eseguire i test che usano i bundle
reali. Servono separatamente gli input e gli artefatti riservati, ottenuti tramite
un canale autorizzato. Non aggiungerli con `git add -f`.

Gli esempi dei test pubblicati usano identificativi fittizi al posto di eventuali
identificativi originali. Gli originali locali e i contratti storici di hash non
sono riscritti: per riprodurre un bundle storico firmato serve l'intero ambiente
riservato coerente con le sue ricevute, non soltanto i dati grezzi.

Lo snapshot di pubblicazione è basato sul `main` remoto e non include i commit
locali storici non pubblicati. Le regole di ignore e l'esclusione dallo snapshot
non rimuovono dati da commit già pubblicati: l'eventuale bonifica della storia
remota è un'attività distinta.

## Struttura attiva

- `data/`: i tre input Parquet e i corrispondenti originali Excel; dati locali non
  distribuiti da Git.
- `src/`, `scripts/`, `configs/`: implementazione, generatori e configurazioni.
- `outputs/`: input analitici e bundle necessari alla catena di riproduzione.
- `astra_metric_audit/notes/phase1_reconciliation.json`: unico input di
  riconciliazione mantenuto di quel vecchio audit, letto dal supervisore del replay.
- `paper/generated/simple_notation_20261020/`: i 13 input TeX e i 5 PDF usati
  dal manoscritto, con manifest che autentica le sorgenti.
- Gli altri pacchetti conservati in `paper/generated/` sono input intermedi
  autenticati dei generatori, non alternative da usare al posto del pacchetto attivo.
- `tests/`: controlli numerici, di ricostruzione, dei generatori e delle figure.
- `history/`: materiale archiviato, escluso da Git. I file conservano il percorso
  originario: ad esempio `docs/a.md` diventa `history/docs/a.md`.

La catena storica autentica anche file non importati direttamente: tutti i moduli
scientifici, diversi script/configurazioni e pacchetti precedenti sono vincolati
tramite hash. Anche `paper/spoofing_new.tex` e il suo PDF sono input protetti di
identità dei generatori, non il manoscritto attivo. Non rimuovere questi file
senza migrare esplicitamente il contratto di provenienza. Le copie ripristinate
di questi due file corrispondono esattamente agli hash delle ricevute storiche;
le versioni presenti prima della pulizia sono conservate in `history/paper/`.
I metadati di un vecchio
pacchetto possono citare analisi archiviate: una citazione storica non equivale a
una dipendenza eseguita dal workflow attivo.

## Ambiente e controlli rapidi

Ambiente locale verificato: `/home/danielemdn/miniconda3/envs/main/bin/python`.
Le versioni osservate sono in `requirements-reproduction.txt`; LaTeX richiede
`latexmk`, `pdflatex`, `bibtex` e `pdftotext`.

Dalla radice del repository:

```sh
export PYTHONPATH="$PWD:$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/danielemdn/miniconda3/envs/main/bin/python
"$PY" scripts/generate_simple_notation_package.py --check
"$PY" -m pytest -p no:cacheprovider -q
BUILD=$(mktemp -d "${TMPDIR:-$HOME/.cache}/spoofing-paper-build.XXXXXX")
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="$BUILD" spoofing_simple.tex)
```

Impostare `PYTHONPATH` evita che eventuali librerie Python iniettate dal processo
chiamante prevalgano sull'ambiente del progetto. La build temporanea non
sovrascrive il PDF del paper né ricrea intermedi LaTeX nella cartella `paper/`.

## Catena di riproduzione

1. Calibrazione dei pesi H=1 s: script e manifest in
   `outputs/depth_kernel_calibration/20261012_100415_top10_h1/`.
   Errori standard bootstrap e renderer sono in
   `outputs/kernel_se_20261012_103855/` e
   `outputs/kernel_h1_paper_release_20261012_103855/`.
   Gli errori standard iid della frequenza di hit sono un passaggio distinto:
   `scripts/generate_hit_frequency_iid_se.py`.
2. Replay con i pesi H=1: adapter
   `outputs/kernel_h1_paper_release_20261012_103855/adapters/new_kernel.py`;
   sorgenti verificati in `outputs/new_kernel_h1_verified_sources/verified_sources.json`.
   Il recupero Ferrari è conservato separatamente dalla release iniziale.
3. Analisi multiscala: `scripts/study_withdrawal_multiscale.py` con
   `configs/withdrawal_multiscale_v1.json`. I bundle usati sono
   `outputs/withdrawal_multiscale_20261013_v4/{RISANAMENTO,NEXI}/` e
   `outputs/withdrawal_multiscale_20261013_v4_ferrari_extended/FERRARI/`.
   Sommario e diagnostiche della campagna rimangono nel primo percorso.
4. Diagnostiche di prezzo: `scripts/study_primary_kernel_prices.py`,
   `scripts/refine_primary_price_signs.py`, quindi
   `scripts/study_ranked_candidate_prices.py`. Input attivo per tutti i candidati:
   `outputs/ranked_candidate_prices_20261015_101002_v2/`.
5. Tabelle: `scripts/generate_withdrawal_multiscale_manuscript.py`,
   `scripts/generate_market_only_manuscript.py`,
   `scripts/generate_rank_only_manuscript.py`,
   `scripts/summarize_ranked_participants.py`,
   `scripts/generate_participant_manuscript.py`.
6. Figure: `scripts/generate_results_evidence_figures.py`; copertura e
   concentrazione attive provengono dal pacchetto `results_evidence_20261019_v4`,
   prezzi da `results_evidence_20261020_v8`. I manifest conservano le rispettive
   versioni del renderer. `scripts/generate_results_evidence_v4.py` è la copia
   byte-identica del renderer storico recuperato: usarla per rigenerare la v4.
   Il renderer corrente rigenera la v8; non sostituisce retroattivamente la
   ricevuta del renderer storico.
7. Presentazione della notazione: `scripts/generate_simple_notation_package.py`,
   quindi compilazione del manoscritto.

Per le opzioni esatte dei singoli stadi usare `--help` e i manifest/configurazioni
conservati. Ogni nuova esecuzione deve avere una directory nuova: i generatori
rifiutano di sovrascrivere i pacchetti pubblicati. Non eseguire in parallelo i
replay pesanti. Diversi manifest contengono percorsi assoluti di questa macchina:
lo spostamento del repository richiede una migrazione esplicita, non la modifica
silenziosa delle ricevute originali.

### Rigenerazione dei pacchetti da input analitici verificati

Con l'ambiente impostato come sopra, scegliere percorsi nuovi:

```sh
RUN=$(mktemp -d "$PWD/outputs/reproduction.XXXXXX")
"$PY" scripts/generate_rank_only_manuscript.py \
    --output "$PWD/paper/generated/withdrawal_rank_$(basename "$RUN")"
"$PY" scripts/generate_participant_manuscript.py \
    --output "$PWD/paper/generated/withdrawal_rank_participants_$(basename "$RUN")"
"$PY" scripts/generate_results_evidence_v4.py --output "$RUN/results_v4"
"$PY" scripts/generate_results_evidence_figures.py --output "$RUN/results_v8"
"$PY" scripts/generate_hit_frequency_iid_se.py --output-root "$RUN/hit_iid"
"$PY" scripts/generate_simple_notation_package.py --output "$RUN/notation"
```

Questi comandi rigenerano ciascuno stadio dai suoi input pubblicati e autenticati;
non cambiano i percorsi del paper né promuovono automaticamente i nuovi output.
La rigenerazione verificata riproduce valori, tabelle e contenuti grafici. Il solo
CSV di composizione può avere un diverso ordine delle righe di categoria:
le righe e i valori sono identici. L'ordine delle execution e dei ranking non
è trattato come intercambiabile.

### Preflight del replay, senza eseguire il detector

Il vecchio `adapters/source_freeze.json` resta una ricevuta storica. Il freeze
esplicito della struttura attuale è `outputs/reproduction_source_freeze.json`:
non sovrascrive quello storico. Il comando seguente è stato verificato su tutti
e tre gli strumenti e non avvia il replay:

```sh
"$PY" outputs/kernel_h1_paper_release_20261012_103855/adapters/new_kernel.py \
    preflight --output-root "$RUN/new_detection" \
    --checkpoint-root "$RUN/new_checkpoints" \
    --freeze "$PWD/outputs/reproduction_source_freeze.json"
```

Il freeze va rinnovato esplicitamente se cambiano sorgenti o configurazioni;
un errore di hash non va risolto disattivando i controlli. Il preflight multiscala
ha inoltre verificato input, kernel, schemi e sorgenti dei tre bundle correnti.

## Archiviazione e limiti della verifica

L'inventario iniziale, gli hash, il piano preciso, il registro degli spostamenti,
i log e l'esito dei controlli sono in `history/_cleanup/`. Nessun commit, staging
o reset è necessario per consultare o ripristinare l'archivio. Git mostrerà come
rimozioni i vecchi percorsi dei file prima tracciati; aggiungere `history/` al
`.gitignore` non cancella la cronologia Git già esistente.

Sono archiviati anche gli audit editoriali basati su snapshot superati. Nei due
moduli di test misti restano attivi i controlli scientifici e delle figure, in
file dedicati; le copie integrali originali restano nell'archivio. Il log iniziale
documenta i quattro fallimenti di quei confronti storici prima della pulizia.
La validazione della pulizia non equivale a una nuova stima completa da dati
grezzi: i replay completi e il bootstrap non sono rilanciati solo per spostare
file. Gli esiti effettivamente verificati sono registrati nella ricevuta finale.
