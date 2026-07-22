---
title: "Call CONSOB — scheda rapida"
subtitle: "Metodo e risultati preliminari sullo screening di sequenze compatibili con spoofing"
date: "22 luglio 2026"
lang: it-IT
papersize: a4
geometry: margin=1.55cm
fontsize: 9.5pt
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
colorlinks: true
header-includes:
  - |
    \usepackage{booktabs}
    \usepackage{xcolor}
    \setlength{\parskip}{2pt}
    \setlength{\parindent}{0pt}
    \renewcommand{\arraystretch}{1.08}
---

> **Messaggio chiave:** pipeline auditabile di *triage*, non classificatore di condotta. Il gate completo seleziona 28 eventi Nexi, tutti nello stesso client-giornata; il risultato richiede revisione umana e non dimostra intento o causalità.

# Apertura — 60 secondi

«Non classifichiamo cancellazioni isolate. Ricostruiamo una sequenza: profilo recente sul lato opposto, esecuzione passiva dello stesso client, ritiro rapido e percorso del prezzo coerente. Confrontiamo poi il ritiro dopo esecuzioni proprie con controlli della stessa sessione. Su 868.149 messaggi, il gate completo individua 28 episodi Nexi, concentrati in un client-giornata, e zero negli altri campioni. È un segnale investigativo circoscritto, non un accertamento.»

# Metodo — sei passaggi

1. Ricostruzione del book visibile top-10 e del ciclo di vita degli ordini.
2. Consolidamento dei child fill in cluster di esecuzione entro 100 ms.
3. Ricerca del profilo dello stesso client sul lato opposto, osservato nei 90 s precedenti.
4. Assegnazione univoca di cancellazioni/riduzioni entro 2 s dopo il fill.
5. Confronto auto-controllato con ancore di altri client, matched su covariate pre-evento.
6. Gate completo: eccesso di ritiro + ritiro rapido + quantità ritirata maggiore del fill + movimento favorevole + reversione a 2 s.

# Numeri da ricordare

| Misura | Risanamento | Nexi | Ferrari |
|---|---:|---:|---:|
| Messaggi input | 143.018 | 243.333 | 481.798 |
| Cluster di esecuzione | 14.257 | 4.738 | 3.153 |
| Cluster con ritiro attribuito | 44 | 365 | 248 |
| Coppie trattato–controllo | 954 | 1.443 | 1.012 |
| Tasso ritiro trattato | 3,98% | 22,52% | 20,26% |
| Tasso ritiro controllo | 3,77% | 20,30% | 19,66% |
| Unità positive, p non aggiustato ≤ 5% | 0 | 1 | 0 |
| **Eventi gate completo** | **0** | **28** | **0** |

## Focus Nexi

- client anonimizzato 23523; una sola partizione del 23 aprile 2024;
- 691 coppie: 218 ritiri trattati contro 167 controlli;
- differenza +7,38 punti percentuali;
- p non aggiustato 0,000676;
- dopo 194 screen: Bonferroni e Benjamini–Hochberg ≈ 0,131;
- 28 eventi completi tra 09:00:50 e 16:03:03: 20 bid, 8 ask;
- rapporto ritiro/fill mediano 2,48; movimento favorevole mediano 1,75 tick; reversione mediana 0,50 tick.

# Cosa dire / cosa non dire

| Dire | Evitare |
|---|---|
| «28 sequenze compatibili, da revisionare» | «28 casi di spoofing» |
| «p-value esplorativo non aggiustato» | «il p-value prova la manipolazione» |
| «concentrazione in un client-giornata» | «Nexi è più manipolata» |
| «coerenza temporale e direzionale» | «gli ordini hanno causato il prezzo» |
| «assenza di segnali completi nel campione» | «assenza di spoofing» |

# Quattro limiti da anticipare

1. Due strumenti coprono un solo giorno; i campioni non sono comparabili.
2. Identificativo client assente in molti messaggi; solo 21/44, 281/365 e 176/248 cluster matched sono attribuibili a client noti.
3. Il risultato Nexi non supera la correzione per 194 screen; le 691 coppie possono essere temporalmente dipendenti.
4. Matching senza caliper: per l'unità positiva la distanza temporale mediana trattato–controllo è 2 h 11 min, il 95° percentile 7 h 11 min.

# Call da 15 minuti

- **0–2 min:** obiettivo e distinzione market making/spoofing.
- **2–6 min:** pipeline in sei passaggi.
- **6–10 min:** funnel e focus Nexi.
- **10–13 min:** limiti, soprattutto molteplicità e matching.
- **13–15 min:** chiedere feedback su identità client, dossier evento e validazione.

# Tre richieste a CONSOB

1. Conferma della semantica e delle assenze di `NMSC_ORIGINALCLIENTIDSHORTCODE`.
2. Indicazione degli elementi minimi del dossier event-level utile alla vigilanza.
3. Possibilità di validare su casi adjudicati o criteri investigativi indipendenti.

# Chiusura

«La pipeline offre tracciabilità e riduzione dell'universo investigativo. Il segnale Nexi merita un dossier event-by-event, ma i dati attuali non giustificano una conclusione sull'intento. Il passo successivo è migliorare matching e robustezza, e validare il formato con l'esperienza di vigilanza.»
