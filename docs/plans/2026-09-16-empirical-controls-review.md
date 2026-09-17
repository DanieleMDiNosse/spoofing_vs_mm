# Revisione dei controlli empirici e proposta ridotta

**Data:** 2026-09-16. **Stato:** report e revisione del disegno; implementazione non autorizzata da questo documento.

> **Proposta recepita nel piano:** su richiesta dell'utente è stato redatto il [piano operativo v2](2026-09-16-empirical-controls-v2-light.md), che prevale per attività, confini e criteri di accettazione. Questo report conserva la review e le motivazioni; i riferimenti al piano del 9 settembre riguardano la v1 ora archiviata. Non sono stati avviati implementazione o test.

**Perimetro della revisione:** lettura del piano del 9 settembre, codice/config correnti, passaggi pertinenti del paper, metadati e cronologia dell'interruzione. Nessun test, benchmark, replay, preflight del runner o calcolo empirico eseguito. I rilievi sul codice sono statici; i tempi storici non sono misure nuove.

## 1. Verdetto

La domanda è sensata, ma il piano precedente è troppo esteso per una prima integrazione nel paper. La domanda difendibile è: **il ritiro della liquidità eleggibile dello stesso attore è più intenso subito dopo una sua esecuzione sul lato opposto rispetto al suo tempo a rischio non esposto?**

Questa verifica aggiunge un confronto mancante alla ricostruzione descrittiva. Non valida l'intero detector: non dimostra intento, causalità, specificità rispetto al market making, precision, recall o false-positive rate. Un market maker che aggiorna le quote dopo un'esecuzione può produrre la stessa associazione. Un esito positivo non valida automaticamente né WMSCI né la congiunzione dei quattro gate.

**Raccomandazione:** mantenere il tempo a rischio completo, ma ridurre la prima versione a un confronto descrittivo entro attore e a una piccola analisi di sensibilità temporale. Rimuovere dal percorso ordinario il ricalcolo integrale del detector e la simulazione Monte Carlo. Il rischio va ricostruito una sola volta con il replay canonico, senza produrre un secondo detector.

Non è possibile essere «totalmente sicuri» di correttezza implementativa e fattibilità senza le future verifiche. Questa revisione stabilisce un disegno più circoscritto e individua dove spendere il budget computazionale; non certifica risultati non eseguiti.

## 2. Rilievi scientifici

### 2.1 Obiettivo giusto, promessa da ridurre

Il paper dichiara esplicitamente l'assenza del confronto con tempi comparabili (`paper/spoofing_new.tex:1336–1339,1488–1497`). Il piano propone correttamente un endpoint indipendente dal gate e include posture senza esecuzioni (`2026-09-09-empirical-negative-controls.md:44–69,95–108`). Questi requisiti vanno conservati.

La validazione della sequenza completa, il confronto con liquidità sicuramente legittima e l'identificazione causale sono problemi diversi. Non vanno aggiunti implicitamente per rendere il controllo più convincente.

### 2.2 L'endpoint misura una postura dinamica, non la coorte congelata del detector

Il rischio attore-lato ammette nuovi ordini eleggibili e rientri nel top-N. Il detector congela invece i candidati prima dell'esecuzione. Le due popolazioni non coincidono. Il nuovo risultato deve essere chiamato **intensità di cancellazione della liquidità eleggibile**, non probabilità di ritiro del candidato originario o validazione dell'episodio strict.

Il numero di cancellazioni per postura-secondo dipende anche dal numero di ordini: una postura con molti ordini offre più occasioni di cancellazione. Mostrare count, quantità ed età del profilo nei gruppi confrontati; se sono molto diversi, non attribuire la differenza al solo timing. La quantità ritirata per secondo può restare una diagnostica secondaria entro strumento, non un nuovo endpoint primario.

### 2.3 Il placebo corrente è restrittivo e non è un test esatto

`negative_controls.py:484–544` richiede che ogni ancora traslata cada esattamente all'inizio di un intervallo osservato, preservando tutti gli intervalli tra esecuzioni. Richiede inoltre lo stesso spell ininterrotto dalla prima all'ultima ancora (`458–481`).

Con clock irregolari e più esecuzioni per blocco, il supporto potrebbe essere minimo. Non lo abbiamo misurato in questa revisione. La sopravvivenza fino all'ultima ancora è inoltre informazione futura rispetto alle prime: il riferimento viene condizionato a posture persistenti. È legittimo descriverlo come riferimento condizionato, non come controllo generalmente indipendente dal futuro.

Anche i blocchi senza traslazioni valide sono ricalcolati con schedule assente (`725–774`). Il report deve distinguere «nessun supporto» da «ritiro nullo» e non confrontare indiscriminatamente il campione osservato completo con un campione placebo ridotto. Moltiplicare i draw non risolve questo problema.

### 2.4 I contrasti aggregati non garantiscono confrontabilità

Il codice emette righe per attore, ma il contrasto aggregato somma numeratori e denominatori tra attori e modalità prima della sottrazione (`negative_controls.py:567–639`). Attori molto attivi possono dominare esposizione e riferimento in proporzioni diverse. Inoltre un unico contrasto `own_only` non sostituisce risultati separati passive/aggressive.

Usare confronti entro attore-giorno-lato e fascia oraria. Gli attori senza alcuna esposizione propria restano nell'audit della popolazione, ma non possono essere presentati come il riferimento entro-attore di chi esegue. I profili di balance sono diagnostiche, non aggiustamento statistico né prova di supporto congiunto sufficiente.

### 2.5 Prima di riutilizzare l'intersezione, separare eventi puntuali e durate

In `withdrawal_risk.py:1044–1083`, la maschera temporale viene estesa con le finestre attive ai ritiri sul confine; il `contrast_label` così ottenuto etichetta poi l'intera durata del segmento. Se un fill e una cancellazione successiva hanno lo stesso timestamp finale, il ritiro può essere post-fill in ordine feed senza rendere post-fill il tempo precedente.

**Correzione richiesta nella futura implementazione:** il denominatore deve dipendere soltanto da rischio ed esposizione, mai dalla presenza di un ritiro. Attribuire separatamente il punto-cancellazione con la regola di ordine canonico. Non spostare retroattivamente una durata per evitare un denominatore nullo. Questo è un rilievo statico del percorso, non una riproduzione eseguita.

### 2.6 Clock e dipendenza restano limiti reali

La configurazione include un giorno FERRARI, un giorno NEXI e 130 date RISANAMENTO. Non sono tre repliche omogenee né centinaia di migliaia di osservazioni indipendenti. I due titoli con un giorno supportano descrizioni di quel giorno, non inferenza tra giorni.

La cronologia e l'audit storico documentano regressioni del clock; il codice corrente ha già una policy di quarantena. Non è corretto ripetere il vecchio finding «manca la policy»: resta da verificare quanta copertura utilizzabile sopravvive nel nuovo percorso. Non risolvere il problema ordinando diversamente i messaggi o cambiando timestamp senza una decisione scientifica esplicita. Le attuali etichette opening/continuous/closing non provano da sole timezone o stato di negoziazione.

## 3. Dove nasce il costo

| Evidenza corrente | Costo o rischio | Intervento proposto |
|---|---|---|
| Runner `826–873,924–959,1653–1659` | Ricalcola tutte le metriche canoniche e confronta numerosi output prima dei controlli; conserva risultati di più strumenti | Separare certificazione della baseline e analisi dei controlli; riusare artefatti congelati verificati |
| `replay_observation.py:123–128,157–174` | Copia tutti gli ordini attivi sia pre sia post evento | Osservatore dedicato con proiezione minima, senza copie complete del book |
| `withdrawal_risk.py:362–388,604–636` | Emette un intervallo per ogni attore-lato vivo a ogni avanzamento del feed, anche se non è cambiato | Intervalli per cambiamento effettivo del profilo, non prodotto messaggi × attori |
| Config `47`; `negative_controls.py:721–780` | 1000 draw, intersezioni ripetute e liste di schedule/statistiche mantenute fino alla fine | Prima versione con quattro traslazioni fisse; eventuale Monte Carlo in estensione separata |
| Runner `1138–1204` | Per ogni segmento scorre di nuovo eventi e quote; la media dei rendimenti è ricalcolata dentro la somma dei residui | Join as-of/ricerca binaria e rolling calcolato una volta; media una volta, non per residuo |
| Runner `1323–1333,1427–1476` | Accumula le tabelle di tutti gli strumenti prima della scrittura finale | Scrittura a chunk e checkpoint per partizione-giorno; report da aggregati |

La cronologia del pilot segnala molte ore, forte uso di RAM/swap e nessun bundle pubblicato. Gli artefatti parziali sotto `outputs/spoofing_empirical_controls` sono stati rimossi e la directory oggi non esiste. Non attribuisco una percentuale del tempo a ciascun collo di bottiglia: manca una profilazione finale utilizzabile qui.

**Ridurre soltanto 1000 draw o aumentare lo swap non è la soluzione:** il ricalcolo canonico oneroso avviene prima dei placebo. Viceversa non è corretto dire che il LOB venga rieseguito per ogni draw: il loop attuale ricalcola le intersezioni sul rischio già costruito.

## 4. Piano rivisto: versione `empirical_controls_v2_light`

Nome proposto, non valore già supportato dal runner. Il JSON corrente resta v1 e non deve essere eseguito pensando che implementi questo report.

### 4.1 Mantenere

- Tutte le posture attribuibili, visibili, nel top-10 e con policy di età coerente con la baseline a 90 secondi; nessuna selezione su cancellazione, FPM, REV, WMSCI o strict.
- Tempo di postura attore-lato, non somma delle durate dei suoi ordini.
- Finestra post-esecuzione di 2 secondi dalla fine dell'esecuzione aggregata; ordini del lato opposto.
- Risultati separati per strumento e livello di identità; modalità passive/aggressive esplicite.
- Identità canoniche, lifecycle, ordine feed, quarantena, scadenza età tra messaggi, confini e cancellazioni fisiche uniche per evento.
- Attori con sola liquidità e nessuna esecuzione nell'audit del rischio.
- Un esito nullo o contrario all'atteso come risultato valido, senza ritoccare il disegno.

L'uguaglianza esatta al limite di età deve essere reconciliata con la baseline prima di implementare: il rischio corrente usa intervalli semiaperti. Non basta dire «stessi 90 secondi» se cambia il trattamento dell'evento sul confine.

### 4.2 Controllo principale: confronto entro attore

Per ogni attore, lato postura, strumento, partizione, giorno e fascia fissa di 30 minuti nel clock verificato, classificare il tempo a rischio in quattro stati disgiunti:

1. nessuna finestra propria attiva;
2. solo finestra propria passive;
3. solo finestra propria aggressive;
4. entrambe attive.

Le finestre della stessa modalità si uniscono: sovrapposizioni non moltiplicano il tempo. Lo stato misto resta visibile ma fuori dai due contrasti semplici. Il riferimento significa **nessuna esecuzione propria qualificante osservata e attribuita nei precedenti 2 secondi**, non assenza certa di esecuzioni economiche proprie, assenza di trading di mercato o attore sicuramente legittimo. Non attribuire un'esecuzione o una quantità al riferimento.

Conservare un audit distinto delle esecuzioni non attribuibili e dell'ambiguità client/firm, inclusa la copertura temporale interessata. Un'uguaglianza o differenza tra namespace non risolve l'identità economica (`withdrawal_risk.py:864–872`). Annotare i periodi ambigui separatamente: la classificazione nei quattro stati riguarda soltanto lo schedule attribuito osservato. Un claim sull'assenza reale di esecuzioni proprie richiederebbe risolvere tali ambiguità, e non è un obiettivo di questa versione.

Per gruppo `g` e stato `s` riportare:

`lambda[g,s] = cancellazioni eleggibili[g,s] / secondi a rischio[g,s]`.

I contrasti principali sono `lambda[g,passive_only] - lambda[g,none]` e l'analogo aggressive, soltanto dove entrambi i tempi sono positivi. Riportare sempre numeratore, denominatore, esclusioni e stato misto. Con zero tempo, intensità null; un conteggio positivo con zero tempo richiede audit del clock/supporto e non va sanato con epsilon o durata inventata.

È un unico endpoint descrittivo con due strati prespecificati, non una scelta tra due risultati concorrenti: pubblicare entrambi anche se uno è nullo, contrario alle attese o non stimabile. Non aggregarli per nascondere differenze e non selezionare ex post il ramo più favorevole. Il riferimento eventualmente condiviso non è additivo tra branch.

Tabella primaria a questo grain. Per una sintesi entro strumento/modalità/livello di identità, media delle differenze entro gruppo con pesi proporzionali al minimo dei due tempi confrontati, definiti prima di leggere i conteggi. Mostrare anche dispersione per actor-day e quota di rischio esclusa per assenza di confronto. È una standardizzazione descrittiva sulla popolazione con sovrapposizione, non un effetto causale. Le righe non sono repliche indipendenti.

Balance minimo: quantità e numero di ordini eleggibili, età del profilo, spread, profondità, attività recente. La volatilità trailing può essere mantenuta se costruita una volta con metodo stabile, non tramite scansioni per segmento. Queste quantità descrivono la composizione degli stati; non chiamarle tutte covariate pre-trattamento: dopo un fill alcune possono esserne già influenzate. Non usarle per rivendicare aggiustamento causale. Mostrare valori mancanti e supporto, senza imputare zero.

### 4.3 Falsificazione temporale economica: quattro traslazioni fisse

Usare, prima di qualsiasi esame dei risultati, `delta = -60, -30, +30, +60 secondi`. Sono scelte pragmatiche esterne alla finestra di 2 secondi, non tempi scientificamente calibrati. Non garantiscono la distruzione di ogni dipendenza seriale. Questa è **sensibilità alla sincronizzazione**, non un test di randomizzazione e non una distribuzione nulla.

- Mantenere book, rischio e cancellazioni osservati; traslare solo lo schedule proprio. Passive e aggressive dello stesso attore-lato si spostano insieme, conservando distanze, finestre e sovrapposizioni. Nessun child fill permutato.
- Usare i tempi traslati come marcatori virtuali, non come nuove esecuzioni. Non assegnare loro prezzo, quantità, WMSCI o gate.
- Non richiedere che una pseudo-ancora sia l'inizio esatto di un intervallo o che la postura sopravviva tra ancore. Intersecare invece lo schedule con il rischio osservato; il mancato supporto resta quantificato. È un cambiamento esplicito rispetto alla selezione placebo v1.
- Per evitare confronti dominati dai bordi, costruire un dominio temporale comune all'offset zero e ai quattro offset, interno alla stessa fascia e allo stesso tratto di clock/copertura valido. Ritagliare bordi e gap sulla base del clock e degli offset, **non** dei ritiri o della persistenza dello spell. Ricostruire anche il risultato a offset zero su quel dominio. Lo schedule traslato si consulta come `E(t-delta)`; non serve trovare un messaggio esistente al nuovo timestamp.
- Nessun wrapping e nessun trasferimento tra giorni, partizioni, fasce o gap. Rischio assente non è gap di osservazione: i due casi devono restare distinti.
- Per il solo grafico di traslazione adottare la stessa convenzione puntuale per tutti gli offset, zero compreso: escludere cancellazioni esattamente al timestamp di apertura virtuale, includere il limite superiore. Riportare le esclusioni. Il contrasto osservato principale mantiene invece l'ordine feed quando timestamp uguali sono distinguibili. Non inventare un `sort_index` per ancore virtuali.
- Ricalcolare numeratori e tempi per ogni offset, mantenendo distinta la componente passive, aggressive e mista. Non copiare la statistica osservata e non confrontare un offset ristretto con lo zero sul campione completo.
- Confrontare le curve sugli stessi gruppi con tempi positivi nei due stati a tutti gli offset, usando pesi fissati dall'offset zero; mostrare anche la tabella completa e il supporto perso. La selezione sul supporto osservato rende anche questa sintesi condizionata, non indipendente dal futuro. Se il supporto comune è insufficiente, dichiarare la falsificazione non informativa, senza cercare offset più favorevoli.

Il risultato utile è capire se il picco è localizzato vicino all'esecuzione o persiste anche dopo lo spostamento. Un picco locale è compatibile tanto con spoofing quanto con quote/inventory management; l'assenza del picco indebolisce l'interpretazione temporale, non prova l'assenza di manipolazione.

### 4.4 Rinviare esplicitamente

| Fuori dalla prima versione | Motivo |
|---|---|
| Confronto principale own vs other | Identità client/firm, deduplicazione trade e composizione del mercato introducono un altro disegno; non certifica market maker legittimi |
| 1000 placebo casuali e relativi schedule completi | Non necessari per una diagnostica descrittiva; supporto e scambiabilità vanno prima ridisegnati |
| Wrong-side, matching 1:1/McNemar | Rischio di tautologia o dipendenza; nessun guadagno necessario per la domanda scelta |
| Rescoring FPM/REV/WMSCI/strict alle pseudo-ancore | Endpoint non definiti senza vere esecuzioni; estraneo al controllo temporale |
| Link completo rischio–episodi come output obbligatorio | Utile a un dossier, non necessario per il rapporto conteggio/tempo; conservare chiavi per eventuale linkage successivo |
| Hazard model, effetti causali e classificazione | Richiedono ipotesi, dati e verifica diversi |
| Inferenza confermatoria e correzioni per molti test | La v2 proposta presenta effetti e sensibilità descrittivi, senza p-value |

Se in seguito servono veri test inferenziali, definire prima l'ipotesi nulla e la legge di ricampionamento, poi la dipendenza per giorno/attore e la molteplicità. Non ribattezzare i quattro offset come test statistico significativo.

## 5. Implementazione computazionalmente leggera

### A. Separare la certificazione della baseline dai controlli

Le tabelle canoniche `execution_metrics.parquet` e `execution_cluster_members.parquet` esistono e sono molto più piccole dell'espansione degli stati. Riusare tutte le esecuzioni qualificanti, non soltanto quelle matched/strict. Verificare hash, schema, popolazione, quantità, ruoli, date, clock e provenienza prima del consumo. Gli hash attestano identità del file, non correttezza economica.

**Variazione del contratto v1:** non ricalcolare FPM/REV, episodi e tutte le metriche a ogni esecuzione dei controlli. Una certificazione esistente può essere riusata solo se copre esattamente sorgenti, parametri, codice e semantica rilevanti. Se manca, resta una verifica separata da completare, non una garanzia che questo report inventa. Ogni modifica al replay condiviso richiede futura regressione/equivalenza prima di riusare quel replay. La mera esistenza di una baseline non prova l'equivalenza con il dirty tree corrente.

### B. Un passaggio rischio per partizione-giorno

Riutilizzare `panel.replay_events` e i suoi hook. Il runner corrente passa invece da snapshot completi: non basta chiamare `compute_withdrawal_risk` e aspettarsi un algoritmo nuovo.

L'osservatore dedicato legge lo stato live senza modificarlo o conservarne riferimenti. Conserva soltanto chiavi lifecycle, identità, lato, quantità, prezzo/rank e tempo di origine necessari; intercetta l'ordine cancellato nello stato pre-evento e aggiorna la proiezione eleggibile nel post-evento. Tutto il book continua a essere ricostruito, inclusi ordini fuori top-10 che potranno rientrare; si riduce ciò che l'osservatore copia/emette, non la popolazione di mercato.

Accumulare intervalli per attore-lato soltanto quando cambiano membership/quantità/eleggibilità o termina la copertura. Età e sue soglie si aggiornano analiticamente; usare una coda ordinata delle scadenze. Il trascorrere del tempo o un messaggio di un altro attore non impone una nuova riga per tutti. Una modifica ai livelli di prezzo può però cambiare l'eleggibilità di molti ordini: gestirla esplicitamente.

Iniziare con una proiezione top-N semplice sui hook canonici; un indice incrementale per prezzo è una possibile ottimizzazione successiva solo se la profilazione lo giustifica. Non costruire ora un nuovo motore LOB per promettere complessità lineare. Anche dopo aver rimosso le copie, una scansione degli ordini attivi per messaggio resta un costo da misurare.

La suddivisione giornaliera deve preservare il warm-up/inizializzazione canonici e gli indici di origine. `replay_events` enumera nuovamente gli indici a ogni chiamata (`panel.py:570–575`): il collegamento agli artefatti congelati richiede una mappa esplicita tra indice locale e originale, non la sostituzione silenziosa dell'uno con l'altro. Non iniziare un book vuoto nel mezzo della giornata. Non estendere il rischio con il primo timestamp della partizione successiva.

### C. Intersezioni indicizzate, non espansione del pannello

Per gruppo, conservare array ordinati di intervalli a rischio, finestre proprie e cancellazioni. Unire finestre sovrapposte della stessa modalità e usare sweep-line o ricerche binarie.

Il tempo esposto si ottiene intersecando intervalli; i conteggi si ottengono cercando gli eventi nelle unioni di finestre. Integrali cumulativi del rischio e conteggi cumulativi delle cancellazioni consentono query agli estremi senza scandire ogni riga per ogni finestra. Intersecare passive/aggressive una volta per determinare il misto e ottenere i due stati esclusivi senza doppi conteggi. Conservare gli indici canonici per le uguaglianze di timestamp osservate.

Con `I` intervalli compatti, `C` cancellazioni e `K` finestre, ogni schedule può essere trattato con sweep ordinati proporzionali a `I+C+K` dopo gli ordinamenti, oppure con query logaritmiche sugli indici. Questa stima riguarda l'intersezione, **non** il replay del book o l'intera pipeline. Niente matrice attori × tutti i messaggi × draw.

### D. Memoria e pubblicazione

- Leggere colonne e giorni necessari con Polars lazy; il filtro dopo `read_parquet` non evita la lettura iniziale completa.
- Un solo strumento/partizione-giorno per volta, nessun parallelismo pesante nella prima versione.
- Scaricare buffer a Parquet con limite esplicito; non tenere i chunk in una lista per concatenarli tutti alla fine.
- Salvare checkpoint autocontenuti con hash/config; ricominciare dal primo checkpoint non valido. Nessun riuso solo perché il filename esiste.
- Dopo il primo passaggio non rieseguire il LOB per cambiare offset o generare il report: riusare il pannello compatto certificato.
- Artefatti minimi: manifest/config/provenienza, intervalli compatti e cambi membership, cancellazioni, audit copertura/clock/transizioni, statistiche per gruppo/offset, balance e report. Niente serializzazione di tutti i risultati intermedi per draw.
- Pubblicare il bundle finale soltanto dopo riconciliazione. Non promuovere `LATEST` o modificare paper e baseline automaticamente.

## 6. Sequenza futura e gate di accettazione

1. Approvare il perimetro v2, il significato dei quattro stati, il dominio degli offset e la distinzione da v1. Aggiornare in seguito config/parser/report insieme: gli attuali validator accettano solo v1.
2. Chiudere su fixture i confini scientifici: no-fill, stessa data/identità, sovrapposizioni, eventi allo stesso timestamp, scadenza a 90 secondi, clock regressivo, cancellazione parziale/duplicata e cambio lifecycle. A rischio e schedule fissati, cambiare l'attribuzione dei punti-cancellazione non deve cambiare i tempi esposti. Separatamente, a storia pre-ancora fissata, cambiare il futuro non deve cambiare l'eleggibilità a quell'ancora; può invece modificare legittimamente il rischio futuro.
3. Verificare equivalenza del replay/estrazione e degli aggregati tra rappresentazione di riferimento e compatta su input piccoli; i risultati v2 non devono essere confrontati con v1 come se i contrasti fossero identici.
4. Solo con nuova autorizzazione, pilot circoscritto e misurato: tempo wall, picco RSS, righe per evento, righe compatte, copertura, supporto. Definire prima un limite RAM/tempo coerente con le risorse libere. Arrestare il pilot senza degradare il desktop, anziché aumentare progressivamente swap e cap.
5. Estendere sequenzialmente ai giorni congelati soltanto dopo il pilot. Un eventuale sottocampione deve essere scelto prima degli outcome e dichiarato come popolazione diversa; non scegliere soltanto giorni con segnale favorevole o più facili.
6. Due review future: scientifica e regressione/risorse. Verificare output persisted, non soltanto helper. Nessuna di queste esecuzioni è stata svolta oggi.

**Criterio di consegna:** una tabella di intensità/differenze entro attore, un grafico/tabella della sensibilità temporale, e un audit leggibile di supporto e copertura. Niente nuovo classificatore, niente significatività fabbricata e nessuna promessa di durata non misurata.

## 7. Effetto di questa revisione sui file

La proposta di questo report è stata recepita nel [piano operativo v2](2026-09-16-empirical-controls-v2-light.md). Il [piano originale](2026-09-09-empirical-negative-controls.md) resta disponibile come archivio del disegno più ampio; codice, config v1, test, raw, risultati canonici e paper non vengono modificati. L'aggiornamento documentale non autorizza l'implementazione o l'esecuzione dei controlli.

## 8. Seconda revisione e decisioni finali

Una revisione scientifica indipendente, anch'essa senza esecuzioni o modifiche, ha confermato la necessità di ridurre il perimetro, distinguere associazione da causalità, evitare selezione sulla sopravvivenza multi-ancora e separare i blocchi placebo senza supporto dai negativi. I passaggi di codice pertinenti sono stati ricontrollati localmente. L'audit di identità e l'obbligo di riportare entrambi gli strati sono stati resi più espliciti in questo report.

Il revisore proponeva pseudo-ancore casuali singole e pooling delle modalità come alternativa minima. Non sono adottati qui: il campionamento separato perde la struttura delle raffiche di esecuzioni, mentre il pooling nasconde una distinzione mantenuta dal progetto. Si conservano quindi entrambi gli strati e si preferisce una sensibilità deterministica a costo limitato, rinunciando esplicitamente all'inferenza Monte Carlo. La specifica finale dei quattro offset non è stata esaminata dal revisore indipendente: la sua review non costituisce approvazione integrale della v2. Correttezza eseguibile e prestazioni restano da verificare soltanto dopo autorizzazione.
