# Confronto tra i nuovi risultati e gli alert CONSOB

## Sintesi

L'estensione con identità `client_then_firm` e ancore di esecuzione sia passive sia aggressive migliora nettamente la **copertura degli eventi segnalati da CONSOB**, ma non produce una corrispondenza completa con la regola comportamentale stretta del detector.

Dei 36 alert con orario, le sovrapposizioni nel file CONSOB producono **34 finestre temporali distinte**. In tutte e 34 si osservano attività del soggetto e almeno un cluster di esecuzione; 30 contengono anche un ritiro attribuito compatibile, mentre solo 4 superano tutti i gate stretti del detector. L'alert RISANAMENTO ha solo la data e deve quindi essere valutato separatamente.

| Titolo | Evidenza CONSOB | Esecuzione recuperata | Ritiro attribuito | Sequenza stretta | Interpretazione |
|---|---:|---:|---:|---:|---|
| FERRARI | 35 alert → 33 finestre distinte | 33/33 | 30/33 | 4/33 | Copertura temporale elevata; la maggior parte non soddisfa tutti i gate di prezzo e reversione. |
| NEXI | 1 finestra | 1/1 | 0/1 | 0/1 | Recuperato un cluster **aggressivo** di 4 fill, ma senza il ritiro attribuito richiesto. |
| RISANAMENTO | 1 giornata, senza orario | 59 cluster nella giornata | 1 cluster nella giornata | 0 | Non è possibile stabilire il richiamo dell'evento specifico senza una finestra intraday. |

## Effetto delle nuove dimensioni

I cluster con ritiro attribuito nel nuovo run si distribuiscono così:

| Titolo | Client/passiva | Client/aggressiva | Firm/passiva | Firm/aggressiva | Totale | Stretti |
|---|---:|---:|---:|---:|---:|---:|
| FERRARI | 238 | 211 | 1.098 | 1.036 | 2.583 | 378 |
| NEXI | 353 | 177 | 711 | 444 | 1.685 | 231 |
| RISANAMENTO | 42 | 57 | 92 | 14 | 205 | 12 |

Il livello firm produce la quota maggiore dei cluster con ritiro attribuito per FERRARI e NEXI. Anche il ramo aggressivo non è accessorio: nel caso NEXI segnalato da CONSOB è ciò che consente di recuperare l'esecuzione osservata.

## Mismatch principali

- **FERRARI:** i 32 alert del soggetto principale diventano 30 finestre non sovrapposte; tutte contengono esecuzioni, 29 hanno un ritiro attribuito e 4 sono strette. Tuttavia l'identità esterna a livello firm corrisponde a due chiavi interne del detector: le 4 corrispondenze sono quindi valide a livello di soggetto/firm, non come richiamo “exact actor”. Per il secondo soggetto, le 3 finestre hanno identità allineata: tutte recuperano esecuzioni, una ha un ritiro attribuito, nessuna è stretta.
- **NEXI:** identità e finestra sono allineate, ma manca il ritiro attribuito entro la regola temporale adottata; l'aggressive execution risolve la copertura dell'esecuzione, non il mismatch comportamentale.
- **RISANAMENTO:** nella giornata risultano 59 cluster del soggetto, di cui 32 aggressivi e 27 passivi, ma l'assenza dell'orario CONSOB impedisce un confronto evento-per-evento.

## Conclusione

Nel run esteso, tutte le finestre temporizzate CONSOB risultano **osservabili** almeno a livello di attività ed esecuzione. Rimane però una differenza tra la logica dell'alert esterno e la definizione stretta usata qui. I 4 match stretti non vanno interpretati come una stima validata del recall: tra le 4 finestre con granularità d'identità perfettamente allineata, nessuna supera tutti i gate. Servono finestre CONSOB precise e casi negativi per calibrare soglie e falsi positivi.

## Provenienza

- Fonte esterna: `Alert eventi layering-2.pdf`.
- Nuovo run: `outputs/spoofing_metrics/20260728_173801_actor_execution_anchor_v2_dashboard_refresh` (`client_then_firm`, anchor `passive,aggressive`, finestra ritiro 2 s, età massima ordine 90 s).
- Confronto riprodotto con `scripts/audit_external_alert_detector_overlap.py`; le 35 segnalazioni FERRARI sono state unite in 33 finestre distinte prima del calcolo, per evitare doppi conteggi.
