---
title: "Screening di sequenze compatibili con spoofing"
subtitle: "Metodo, risultati preliminari e punti di discussione con CONSOB"
author: "Nota di lavoro per la call del 22 luglio 2026"
date: "21 luglio 2026"
lang: it-IT
papersize: a4
geometry: margin=1.9cm
fontsize: 10pt
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: "1F4E79"
urlcolor: "1F4E79"
toc: true
toc-title: "Indice"
toc-depth: 2
header-includes:
  - |
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{array}
    \usepackage{xcolor}
    \definecolor{consobblue}{HTML}{1F4E79}
    \definecolor{softgray}{HTML}{F3F5F7}
    \setlength{\parskip}{4pt}
    \setlength{\parindent}{0pt}
    \setlength{\emergencystretch}{3em}
    \renewcommand{\arraystretch}{1.16}
---

> **Stato del documento — risultati preliminari, uso di lavoro.**  
> Il metodo produce indicatori di *screening* e dossier verificabili. Non attribuisce intenzionalità, non stima la prevalenza dello spoofing e non costituisce da solo evidenza di una violazione.

# Come usare questo report

Questo documento serve a due scopi:

1. offrire una spiegazione sufficientemente dettagliata del metodo senza entrare nei passaggi matematici non necessari alla call;
2. mantenere una separazione netta tra ciò che è stato osservato, ciò che è un'interpretazione economica e ciò che deve ancora essere validato.

Per una call di circa 15 minuti, è sufficiente seguire le sezioni **Messaggio centrale**, **Metodo in sei passaggi**, **Risultati** e **Limiti**. Le sezioni finali contengono una scaletta temporizzata, risposte a domande probabili e la provenienza dei numeri.

# Messaggio centrale

## In una frase

Abbiamo costruito una pipeline auditabile che ricostruisce il book visibile e cerca **sequenze di ordini, esecuzioni e cancellazioni compatibili con il meccanismo dello spoofing**; sui tre campioni disponibili emerge un nucleo ristretto di 28 episodi Nexi meritevoli di revisione, ma la concentrazione in un solo client-giornata e i limiti statistici e informativi non consentono ancora conclusioni confermative.

## Quattro punti da far arrivare subito

1. **Non classifichiamo singoli messaggi.** Ricostruiamo una sequenza economica: liquidità sul lato opposto, esecuzione del partecipante, ritiro rapido e andamento dei prezzi coerente con l'ipotesi.
2. **Il confronto è intra-giornata e auto-controllato.** Per ogni client confrontiamo ciò che accade dopo le sue esecuzioni con ancore di controllo costruite su esecuzioni di altri partecipanti, usando per il matching solo informazioni disponibili prima dell'evento.
3. **Il risultato principale è selettivo, non diffuso.** Il gate completo trattiene 28 eventi Nexi e zero negli altri due campioni. I 28 eventi appartengono tutti allo stesso client anonimizzato e alla stessa giornata.
4. **È un risultato di triage.** Il sistema ordina l'attenzione investigativa e conserva la provenienza di ogni legame; non dimostra l'intento, l'effetto causale sul prezzo o una violazione.

## Apertura consigliata della call — 60 secondi

> «L'obiettivo non è assegnare automaticamente un'etichetta di spoofing, ma ridurre in modo tracciabile l'universo degli eventi da esaminare. Partiamo dai messaggi ordine per ordine, ricostruiamo il book visibile e cerchiamo una sequenza precisa: un profilo recente sul lato opposto, un'esecuzione passiva dello stesso client, il rapido ritiro di quel profilo e una dinamica di prezzo coerente. Confrontiamo poi la frequenza del ritiro dopo le esecuzioni proprie con controlli della stessa sessione. Sui campioni attuali il gate completo individua 28 episodi Nexi, tutti concentrati in un client-giornata; Risanamento e Ferrari non producono episodi completi. È quindi un segnale investigativo circoscritto, non una conclusione sull'intenzionalità.»

# Perché il problema richiede una sequenza

Cancellare molti ordini o mostrare grandi quantità non è di per sé anomalo. Un market maker può aggiornare frequentemente le quotazioni, ridurre l'esposizione per rischio di inventario o reagire a informazione e selezione avversa. Di conseguenza:

- una soglia di quantità isolata genera facilmente falsi positivi;
- un elevato rapporto ordini/eseguiti non identifica il motivo delle cancellazioni;
- la sola asimmetria del book non collega il profilo di liquidità a una successiva esecuzione favorevole;
- l'intento non è osservato direttamente nei dati.

La domanda operativa diventa quindi più specifica:

> **Un client ritira rapidamente liquidità recente sul lato opposto dopo una propria esecuzione, più spesso di quanto faccia in circostanze simili senza quella propria esecuzione, e il percorso del prezzo è compatibile con il meccanismo ipotizzato?**

Il metodo scompone questa domanda in condizioni osservabili e conserva tutti gli identificativi necessari per tornare ai messaggi originari.

# Dati e perimetro effettivo

## Campioni analizzati

| Strumento | Periodo coperto | Messaggi grezzi | Partizioni di trading | Natura del campione |
|---|---:|---:|---:|---|
| Risanamento | giugno–novembre 2024 | 143.018 | 130 | multi-periodo |
| Nexi | 23 aprile 2024 | 243.333 | 1 | singola giornata |
| Ferrari | 13 giugno 2024 | 481.798 | 1 | singola giornata |
| **Totale** | — | **868.149** | **132** | campioni eterogenei |

Le differenze tra strumenti sono **descrittive**, non confronti controllati: durate, composizione dei partecipanti, liquidità e regime di mercato non sono omogenei.

## Informazione utilizzata

La pipeline usa messaggi order-level per ricostruire:

- stato del book visibile sui primi 10 livelli per lato;
- ciclo di vita degli ordini: inserimento, modifica, fill, cancellazione e altri eventi;
- lato, prezzo, quantità visibile e quantità residua;
- client anonimizzato e trading firm quando disponibili;
- timestamp e ordinamento originario della fonte;
- migliori prezzi, mid-price e microprice quando calcolabili.

## Stato della ricostruzione del book

L'audit sui tre file riporta:

- 868.149 righe normalizzate e riconciliate con gli input;
- tutte le colonne richieste presenti e nessun codice enumerativo ignoto nei campioni;
- nessuna duplicazione della chiave di ordinamento;
- zero stati locked/crossed prima o dopo l'evento nella ricostruzione corrente;
- zero categorie di errore bloccante.

Questi controlli verificano coerenza contabile e alcune invarianti del book, **non l'esatta posizione FIFO in coda**. La ricostruzione è volutamente conservativa per gli ordini marketable: non li inserisce come liquidità passiva visibile quando la struttura multi-riga del feed li farebbe apparire temporaneamente locked/crossed.

Sono inoltre propagate tre classi di avvertimento, non nascoste:

- 17.245 eventi marketable non trattati come liquidità passiva resting;
- 11.820 eventi non prezzati e non resting, prevalentemente market/stop-market;
- 634.273 messaggi senza `NMSC_ORIGINALCLIENTIDSHORTCODE` compilato sulla singola riga.

L'ultimo punto non impedisce la ricostruzione del book, ma limita l'attribuzione participant-level. I casi senza identificativo verificabile sono conteggiati separatamente e non devono essere interpretati come un unico attore economico.

# Metodo in sei passaggi

## 1. Ricostruzione del book e dei partecipanti

I messaggi sono ordinati per chiave temporale e sequenza sorgente. Per ogni evento si aggiorna lo stato del book visibile e si mantiene la storia dell'ordine e del client. Sono esclusi dalla profondità visibile gli eventi non resting o condizionali finché non diventano effettivamente visibili.

**Perché conta:** la misura non si basa sul volume aggregato del mercato, ma sul profilo che il singolo partecipante espone ai diversi livelli del book.

## 2. Consolidamento dei fill in esecuzioni economiche

Un singolo ordine passivo può generare più righe di fill. Le righe vengono riunite in un **cluster di esecuzione** se condividono ordine passivo, client, lato, prezzo e partizione, se i fill consecutivi distano al massimo 100 millisecondi e non vi sono interruzioni del ciclo di vita.

In questo modo l'unità analitica è l'esecuzione economica ricostruita, non ogni messaggio tecnico del feed. L'appartenenza delle righe originarie al cluster resta disponibile per audit.

## 3. Profilo recente sul lato opposto

Per ciascun cluster si cercano ordini visibili dello stesso client sul lato opposto:

- presenti prima dell'esecuzione;
- osservati per la prima volta non più di 90 secondi prima;
- collocati nei primi 10 livelli del book.

Il limite di 90 secondi evita di associare automaticamente all'episodio liquidità molto vecchia e strutturale.

## 4. Ritiro rapido e posteriorità nell'ordinamento

Dopo la fine del cluster si cercano riduzioni o cancellazioni del profilo candidato entro 2 secondi. Una cancellazione è eleggibile solo se segue l'ultimo fill sia nel timestamp sia nell'ordine della fonte.

Ogni cancellazione fisica può essere assegnata a **un solo cluster**, con una regola deterministica. Questo evita di moltiplicare artificialmente la stessa azione su più episodi vicini.

## 5. Confronto auto-controllato

Per ogni unità client–partizione si costruiscono coppie:

- **trattato:** il client ha avuto una propria esecuzione passiva mentre esponeva il profilo candidato;
- **controllo:** un altro client ha avuto un fill nella stessa partizione mentre il client focale esponeva un profilo comparabile, ma senza una propria esecuzione in quell'ancora.

Il controllo viene scelto senza riutilizzare la stessa ancora fisica e usando solo caratteristiche pre-evento: numero di ordini candidati, quantità candidata, età del profilo e vicinanza temporale. Si confronta quindi la probabilità di ritiro entro 2 secondi con un test esatto per dati appaiati.

**Interpretazione corretta:** il confronto riduce alcune differenze stabili tra client, ma non crea da solo un esperimento causale. Attività intraday, volatilità, notizie e qualità del matching possono restare confondenti.

## 6. Gate descrittivo completo

Un evento entra nel dossier finale solo se sono contemporaneamente vere cinque condizioni:

1. l'unità client–partizione mostra un eccesso positivo di ritiro nel confronto appaiato, con soglia non aggiustata del 5%;
2. esiste un ritiro attribuito entro 2 secondi dopo l'esecuzione;
3. la quantità ritirata sul lato opposto supera la quantità eseguita;
4. prima del fill il mid-price si è mosso nella direzione favorevole all'esecuzione;
5. dopo il ritiro il prezzo mostra reversione nella finestra di 2 secondi.

Questo gate non è un punteggio opaco: per ogni riga è possibile vedere quale condizione passa o fallisce.

# Il ruolo del kernel di profondità

I primi 10 livelli non vengono trattati come equivalenti. Per ogni strumento e lato si stima un peso empirico per rango combinando:

- la probabilità che, entro 10 secondi, il miglior prezzo raggiunga quel livello;
- il grado di co-movimento osservato tra liquidità visibile e variazione del mid-price.

I pesi sono normalizzati separatamente per bid e ask. In termini intuitivi, il kernel risponde alla domanda: **quali parti del profilo visibile sono più informative nel campione di quello strumento?**

Il kernel:

- è un input del modello di sorveglianza, non un'etichetta;
- non è interpretato come effetto causale della liquidità sul prezzo;
- è stimato sullo stesso campione e non è ancora validato out-of-sample;
- non è usato per rendere il gate finale dipendente da un punteggio arbitrario.

# Risultati verificati

## Funnel per strumento

| Misura | Risanamento | Nexi | Ferrari |
|---|---:|---:|---:|
| Messaggi di fill passivo grezzi | 15.341 | 5.402 | 3.498 |
| Cluster di esecuzione | 14.257 | 4.738 | 3.153 |
| Cluster con profilo candidato pre-fill | 1.290 | 1.586 | 1.220 |
| Cluster con ritiro attribuito entro 2 s | 44 | 365 | 248 |
| di cui attribuibili a client noto | 21 | 281 | 176 |
| Cancellazioni fisiche assegnate | 46 | 522 | 312 |
| Coppie trattato–controllo | 954 | 1.443 | 1.012 |
| Unità client–partizione valutate | 165 | 17 | 12 |
| Unità con eccesso positivo, p non aggiustato ≤ 5% | 0 | 1 | 0 |
| Eventi che passano il gate completo | **0** | **28** | **0** |

Percentuale di cluster con ritiro attribuito entro 2 secondi:

- Risanamento: **0,31%** dei cluster;
- Nexi: **7,70%**;
- Ferrari: **7,87%**.

Questi tassi non sono prevalenza dello spoofing. In particolare, il matching meccanico include anche casi non attribuibili a un client noto: la quota attribuibile è 47,7% per Risanamento, 77,0% per Nexi e 71,0% per Ferrari.

## Confronto aggregato trattato–controllo

| Strumento | Coppie | Ritiri dopo esecuzione propria | Ritiri nei controlli | Differenza |
|---|---:|---:|---:|---:|
| Risanamento | 954 | 38 / 954 = 3,98% | 36 / 954 = 3,77% | +0,21 punti percentuali |
| Nexi | 1.443 | 325 / 1.443 = 22,52% | 293 / 1.443 = 20,30% | +2,22 punti percentuali |
| Ferrari | 1.012 | 205 / 1.012 = 20,26% | 199 / 1.012 = 19,66% | +0,59 punti percentuali |

Questa tabella aggrega unità molto diverse ed è solo descrittiva. Il test viene eseguito a livello client–partizione, non sul totale dello strumento.

## Unità Nexi che supera lo screen non aggiustato

Una sola unità, riferita al client anonimizzato **23523** nella partizione del 23 aprile 2024, supera lo screen positivo al 5% non aggiustato:

- 691 coppie appaiate;
- 218 ritiri su 691 ancore trattate: **31,55%**;
- 167 ritiri su 691 controlli: **24,17%**;
- differenza: **+7,38 punti percentuali**;
- coppie discordanti favorevoli: 148;
- coppie discordanti avverse: 97;
- p-value esatto unilaterale non aggiustato: **0,000676**.

### Cautela statistica essenziale

Lo screen è stato applicato a **194 unità client–partizione** complessive. Un audit aggiuntivo per questo report produce, per l'unità Nexi:

- p-value corretto Bonferroni: **0,131**;
- q-value Benjamini–Hochberg: **0,131**.

Il risultato quindi **non rimane sotto il 5% dopo correzione per molteplicità**. Inoltre, le 691 coppie appartengono alla stessa sessione e possono presentare dipendenza temporale; il test esatto standard non risolve questa dipendenza. Il p-value va presentato come indicatore esplorativo, non come conferma statistica definitiva.

## I 28 eventi Nexi del gate completo

I 28 eventi finali:

- appartengono tutti al client anonimizzato 23523 e alla stessa partizione;
- si distribuiscono tra le 09:00:50 e le 16:03:03;
- comprendono 20 esecuzioni sul bid e 8 sull'ask;
- rappresentano l'11,1% dei 252 cluster con ritiro attribuito per quel client;
- rappresentano lo 0,59% dei 4.738 cluster Nexi complessivi.

Statistiche puramente descrittive dei 28 eventi selezionati:

- quantità eseguita mediana: 370 unità, intervallo 1–1.280;
- rapporto mediano tra quantità ritirata e quantità eseguita: 2,48;
- movimento favorevole pre-fill mediano: 1,75 tick;
- reversione post-cancellazione mediana: 0,50 tick.

Queste statistiche sono condizionate dal gate: movimento favorevole e reversione sono positivi **per costruzione**. Non possono quindi essere usati come prova indipendente dell'effetto del profilo sul prezzo.

# Lettura sostanziale dei risultati

## Ciò che i risultati mostrano

- La pipeline riesce a ridurre un universo di 868.149 messaggi a episodi ricostruibili, con chiavi e condizioni verificabili.
- Il ritiro rapido dopo fill è osservabile su tutti gli strumenti, ma il confronto con controlli simili è generalmente vicino alla parità.
- Solo una unità Nexi mostra un eccesso marcato di ritiro nel test non aggiustato.
- Il gate economico più restrittivo concentra ulteriormente l'evidenza in 28 eventi dello stesso client-giornata.
- La concentrazione è utile per il triage: suggerisce di esaminare un dossier coerente anziché centinaia di cancellazioni isolate.

## Ciò che i risultati non mostrano

- Non mostrano che i 28 eventi siano 28 violazioni o 28 tentativi indipendenti.
- Non provano che il profilo di ordini abbia causato il movimento del prezzo.
- Non rivelano l'intento del partecipante.
- Non stimano la frequenza dello spoofing nei tre titoli o nel mercato.
- Non consentono di concludere che Nexi sia «più manipolata» di Risanamento o Ferrari.
- Non escludono spiegazioni di market making, gestione dell'inventario, aggiornamento di fair value o reazione a informazione.

## Interpretazione più prudente

La formulazione consigliata è:

> «Nel campione Nexi del 23 aprile 2024, un client anonimizzato presenta un insieme concentrato di sequenze che soddisfano congiuntamente criteri temporali, di quantità e di percorso del prezzo compatibili con il meccanismo ipotizzato. Il risultato giustifica una revisione event-by-event e analisi di robustezza; non è sufficiente per inferire intenzionalità manipolativa.»

# Limiti da dichiarare prima che vengano chiesti

## 1. Perimetro ridotto ed eterogeneo

Due campioni coprono una sola giornata; Risanamento copre un periodo più lungo. Non esiste ancora una base omogenea per confronto tra titoli, periodi e regimi di volatilità.

## 2. Copertura degli identificativi client

Il codice client originario non è compilato in ogni messaggio. La storia ordine può propagare l'identità quando disponibile, ma una quota dei cluster matched resta non attribuibile. Questi eventi non vanno trasformati in conclusioni participant-level.

## 3. Molteplicità e dipendenza temporale

Lo screen corrente usa una soglia del 5% non aggiustata. Il solo risultato positivo non supera una correzione semplice sulle 194 unità. Le coppie della stessa giornata possono inoltre essere serialmente dipendenti.

## 4. Qualità del matching

Per l'unità Nexi positiva, il numero di ordini candidati coincide esattamente nell'88,6% delle 691 coppie, ma non sono ancora imposti caliper di qualità. La distanza temporale mediana tra trattato e controllo è circa **2 ore e 11 minuti**; il 95° percentile è circa **7 ore e 11 minuti**. Questo può lasciare differenze di regime intraday non controllate.

## 5. Sensibilità dei parametri non ancora riportata

Le conclusioni sono condizionate a scelte operative:

- 100 ms per aggregare child fill;
- 90 s di età massima del profilo candidato;
- 2 s per il ritiro rapido;
- 2 s per la reversione;
- 10 livelli e 10 s per la struttura del kernel.

Non è ancora disponibile una mappa di robustezza sistematica a finestre alternative.

## 6. Ricostruzione visibile, non coda completa

Il book ricostruito rispetta le principali invarianti e conserva la provenienza degli eventi, ma non certifica l'esatta priorità FIFO. La gestione degli ordini marketable è conservativa e deve essere tenuta presente in qualunque interpretazione di profondità o permanenza.

## 7. Nessun ground truth di condotta

Non sono disponibili etichette investigative o giudiziarie per stimare precisione, recall, falsi positivi e falsi negativi. Il gate è quindi una regola descrittiva, non un classificatore validato.

## 8. Kernel stimato in-sample

I pesi di profondità sono calibrati sugli stessi dati analizzati. Sono utili per descrivere il profilo, ma non costituiscono validazione predittiva fuori campione.

# Prossimi passi proposti

## Priorità 1 — dossier dei 28 episodi Nexi

Per ciascun episodio produrre una scheda con:

- timeline dei messaggi e identificativi sorgente;
- book top-10 prima del profilo, al fill, alla cancellazione e dopo 2 secondi;
- quantità inserite, modificate, eseguite e ritirate;
- ordini del medesimo client sui due lati;
- andamento di best bid, best ask, mid-price e microprice;
- eventuali esecuzioni e cancellazioni concorrenti di altri operatori;
- qualità dati e copertura dell'identità.

Obiettivo: permettere a un analista di confermare o scartare rapidamente l'interpretazione automatica.

## Priorità 2 — robustezza statistica e del matching

- introdurre caliper temporali e di distanza sulle covariate;
- confrontare controlli nella stessa fascia intraday e nello stesso regime di spread/volatilità;
- usare inferenza cluster/blocked o randomizzazione che rispetti la dipendenza temporale;
- controllare formalmente la molteplicità;
- eseguire placebo e negative controls;
- riportare la sensibilità a finestre e soglie alternative.

## Priorità 3 — estensione e validazione esterna

- ampliare il numero di giornate e strumenti con campioni omogenei;
- validare il significato e la propagazione degli identificativi client con il tracciato e con CONSOB;
- ottenere, se possibile, un piccolo insieme di casi adjudicati per valutare falsi positivi e falsi negativi;
- separare calibrazione del kernel e valutazione out-of-sample;
- definire una soglia operativa solo dopo la validazione dei dossier.

# Scaletta consigliata per una call di 15 minuti

| Tempo | Contenuto | Messaggio da lasciare |
|---|---|---|
| 0:00–1:30 | Obiettivo | «Screening auditabile, non etichetta automatica.» |
| 1:30–3:30 | Perché una sequenza | «Dimensione e cancellazione isolate confondono market making e condotta sospetta.» |
| 3:30–6:30 | Pipeline in sei passaggi | Book → cluster fill → profilo recente → ritiro → controllo → gate. |
| 6:30–9:30 | Numeri del funnel | 868.149 messaggi; 44/365/248 cluster matched; 0/28/0 eventi completi. |
| 9:30–11:30 | Focus Nexi | Un client-giornata; 691 coppie; +7,38 pp; 28 eventi da revisionare. |
| 11:30–13:30 | Cautela | Molteplicità, dipendenza, matching intraday, client mancanti, campione limitato. |
| 13:30–15:00 | Richiesta a CONSOB | Feedback sul tracciato identità, soglie operative e formato dossier. |

## Tre domande concrete da rivolgere ai funzionari

1. **Identità e tracciato:** l'uso di `NMSC_ORIGINALCLIENTIDSHORTCODE` e la propagazione sull'ordine sono coerenti con la semantica attesa del feed? Quali assenze sono fisiologiche?
2. **Dossier investigativo:** quali elementi minimi devono comparire in una scheda evento perché sia utile alla revisione di vigilanza?
3. **Validazione:** quali casi storici o criteri di adjudication potrebbero essere usati per misurare falsi positivi e falsi negativi senza incorporare direttamente l'esito nel metodo?

# Domande probabili e risposte brevi

**«Avete trovato 28 casi di spoofing?»**  
No. Abbiamo trovato 28 sequenze Nexi che passano un gate descrittivo compatibile con il meccanismo. Devono essere revisionate; non sono 28 accertamenti.

**«Perché non basta guardare le cancellazioni elevate?»**  
Perché sono tipiche anche del market making. Il metodo richiede collegamento tra profilo recente, esecuzione propria, ritiro condizionato e percorso del prezzo.

**«Il risultato Nexi è statisticamente significativo?»**  
Solo allo screen non aggiustato: p = 0,000676 per una unità. Considerando 194 unità, Bonferroni e Benjamini–Hochberg portano il valore a circa 0,131. Inoltre le coppie sono temporalmente dipendenti. Va trattato come esplorativo.

**«Perché 2 secondi e 90 secondi?»**  
Sono valori operativi coerenti con l'idea di ritiro rapido e profilo recente, ma non ancora soglie definitive. La robustezza a valori alternativi è un prossimo passo necessario.

**«Che cosa distingue i controlli?»**  
Sono ancore nella stessa partizione in cui il client focale espone un profilo simile ma l'esecuzione è di un altro partecipante. Il matching usa solo informazioni pre-evento e ogni controllo fisico è usato una sola volta.

**«I controlli sono perfettamente comparabili?»**  
No. Il matching è deterministico e auditabile, ma non usa ancora caliper; per l'unità positiva alcuni controlli sono distanti diverse ore. Serve un affinamento intraday.

**«Il movimento del prezzo è causato dagli ordini del client?»**  
Non possiamo dirlo. Il gate verifica coerenza temporale e direzionale, non causalità. Altri ordini, notizie e dinamiche di mercato possono spiegare il movimento.

**«Come gestite i client mancanti?»**  
Li segnaliamo esplicitamente e li escludiamo dalle classifiche e dalle conclusioni participant-level. I cluster matched non attribuibili sono riportati separatamente.

**«A cosa serve il kernel top-10?»**  
A non trattare tutti i livelli come ugualmente informativi. I pesi sono stimati per strumento e lato, ma non costituiscono prova di impatto né etichetta.

**«Il book ricostruito è affidabile?»**  
Ha superato controlli di schema, contabilità, ordine e invarianti su tutti i file, con zero stati locked/crossed nella versione corrente. Resta una ricostruzione del book visibile, non una certificazione della coda FIFO completa.

**«Cosa manca per l'uso operativo?»**  
Dossier event-level, robustezza delle finestre, matching con caliper, inferenza temporale e multipla, più giorni e strumenti, verifica con casi adjudicati e procedure di monitoraggio della qualità dati.

# Lessico consigliato

## Formulazioni da usare

- «sequenze compatibili con il meccanismo ipotizzato»;
- «segnale meccanico di screening»;
- «dossier prioritario per revisione umana»;
- «evidenza descrittiva / esplorativa»;
- «ritiro attribuito nell'ordinamento temporale e sorgente»;
- «client anonimizzato»;
- «spiegazioni alternative esplicitamente aperte».

## Formulazioni da evitare

- «abbiamo rilevato 28 spoofing»;
- «il p-value prova la manipolazione»;
- «Nexi è più manipolata degli altri titoli»;
- «gli ordini hanno causato il movimento del prezzo»;
- «l'assenza di eventi in Ferrari/Risanamento prova l'assenza di spoofing»;
- «il modello conosce l'intento del partecipante».

# Provenienza e riproducibilità

## Run empirica di riferimento

Directory della run:

```text
outputs/spoofing_metrics/
20260721_110005_empirical_kernel_top10_h10_age90_timestamp_sort_causality_fix/
```

La run conserva:

- configurazione risolta e snapshot del codice;
- hash SHA-256 degli input, dei kernel e degli artefatti;
- cluster di esecuzione e membership dei child fill;
- candidati fill–cancel, incluse le alternative e l'assegnazione canonica;
- risk set trattato–controllo e indici sorgente per audit temporale;
- riepiloghi client–partizione;
- eventi del gate completo;
- report e tabelle generate per il paper.

Il validatore della run è stato rieseguito il 21 luglio 2026 e ha restituito `validation_status=passed`. Verifica, tra l'altro, hash, normalizzazione del kernel, unicità dei cluster e delle cancellazioni, posteriorità del ritiro nel timestamp e nell'ordine sorgente, risk set 1:1, non riuso dei controlli, ricalcolo indipendente del test esatto e uguaglianza con le tabelle del paper.

Comando di verifica dalla root del repository:

```text
cd outputs/spoofing_metrics
RUN=20260721_110005_empirical_kernel_top10_h10_age90_timestamp_sort_causality_fix
/home/danielemdn/miniconda3/envs/main/bin/python "$RUN/validate_run.py"
```

## Fonti numeriche principali

- in `paper/generated/`: `cluster_summary.csv`, `top_clients.csv`, `metadata.json`;
- nella root della run: `validation_report.json` e `run_manifest.json`;
- nella sottodirectory Nexi della run:
  - `client_session_withdrawal_excess.csv`;
  - `spoofing_compatible_events.csv`;
  - `withdrawal_risk_sets.csv`;
- nell'audit LOB del 18 giugno 2026: `all_files_warning_audit.json`;
- nella calibrazione top-10 del 17 luglio 2026: kernel e report per strumento.

Le correzioni per molteplicità e le statistiche di qualità del matching riportate in questo documento sono calcoli di audit aggiuntivi eseguiti sulle tabelle persistite della run; non modificano gli artefatti originari.

# Chiusura consigliata

> «Il valore attuale del lavoro è la tracciabilità: ogni numero del funnel può essere ricondotto ai messaggi, ogni cancellazione è assegnata una sola volta e ogni condizione del gate è ispezionabile. Il risultato Nexi è sufficientemente concentrato da giustificare un dossier approfondito, ma non sufficientemente robusto da sostenere una conclusione sull'intento. Il confronto con CONSOB può aiutarci soprattutto su semantica degli identificativi, formato del dossier e disegno della validazione.»
