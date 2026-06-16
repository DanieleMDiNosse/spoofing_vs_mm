# Data dictionary for `keep_cols`

Source: `tracciato_database_tabella_ordini.docx`, table *Field / Datatype / Description*.

This note documents the selected columns in

```python
keep_cols = [...]
df_filtered = df.select(keep_cols)
```

The source document uses mostly lower-case database field names. The columns in the Python list are upper-case and, in a few cases, appear to be renamed or decoded columns rather than literal native fields in the specification.

## Important caveats

1. **Columns ending in `(*)` are coded categorical fields.**  
   The document explains their economic meaning, but the complete code-value dictionaries are not included in the provided file. For example, the document says that `orderside` indicates the side of the order, but it does not provide the full numerical mapping such as buy/sell/cross.

2. **Some selected columns are derived indicators.**  
   `ORDERSTATUS`, `PASSIVEORDER`, `AGGRESSIVEORDER`, `DEAINDICATOR`, `INVESTMENTALGOINDICATOR`, and `EXECUTIONALGOINDICATOR` are not listed as native fields with those exact names. They are naturally obtained by decoding bit fields such as `orderqualifiers`, `tradequalifier`, and `mifidindicators`.

3. **`ISIN` is a renamed field.**  
   The source document calls the ISIN field `codisin` and describes it as the ISIN code.

4. **Prices and quantities may need scaling.**  
   The document repeatedly says that prices are to be calculated with *Price/Index Level Decimals* and quantities with *Quantity Decimals*. If the table stores raw exchange values, you may need fields such as `enr_pricedecimals` and `enr_quantitydecimals`, which are not currently included in `keep_cols`. If the database layer already materializes decimal prices/quantities as `number(30,8)`, no further scaling may be necessary. This should be checked empirically.

---

## 1. Instrument / session keys

| Column | Source field in document | Native datatype | Meaning |
|---|---:|---:|---|
| `TRADEDATE` | `tradedate` | `date` | Trading date. This is the natural session/day key. |
| `ISIN` | `codisin` | `varchar2(12 char)` | ISIN code of the instrument. The source document names this field `codisin`; your dataset appears to expose it as `ISIN`. |

---

## 2. Event ordering / timestamps

| Column | Source field in document | Native datatype | Meaning |
|---|---:|---:|---|
| `HDR_APPLKEYSEQUENCENUMBER` | `hdr_applkeysequencenumber` | `number(20)` | Technical sequencing field. The document labels it as technical and does not give an economic interpretation. Useful as an additional ordering key. |
| `HDR_HWMSEQUENCENUMBER` | `hdr_hwmsequencenumber` | `number(20)` | Technical sequencing field. The document labels it as technical. Useful for deterministic ordering and reconciliation. |
| `HDR_OFFSETID` | `hdr_offsetid` | `number(20)` | Technical offset field. The document labels it as technical. It can be useful to break ties or trace the message stream. |
| `SEQUENCETIME` | `sequencetime` | `timestamp(9)` | Unique sequence time set by the Matching Engine and used for synchronization across Kafka topics. Expressed as nanoseconds since `1970-01-01 UTC`. |
| `BOOKIN` | `bookin` | `timestamp(9)` | Matching Engine input time: the time at which the inbound message enters the Matching Engine. Expressed in nanoseconds since `1970-01-01 UTC`. |
| `BOOKOUTTIME` | `bookouttime` | `timestamp(9)` | Matching Engine output time: the time at which the corresponding message leaves the Matching Engine. Expressed in nanoseconds since `1970-01-01 UTC`. |
| `TRADETIME` | `tradetime` | `timestamp(9)` | Time of the trade. The document states that it equals the Matching Engine input time when the aggressor enters the Matching Engine. |
| `ROW_NUMBER` | not found as a native source field | derived / ETL field | Row counter introduced outside the provided source specification. It is probably an ETL convenience variable used for deterministic ordering after extraction. Treat it as dataset-specific rather than exchange-native. |

### Practical ordering note

For event reconstruction, a robust sorting key is usually something like

```python
["TRADEDATE", "ISIN", "SEQUENCETIME", "HDR_APPLKEYSEQUENCENUMBER", "HDR_HWMSEQUENCENUMBER", "HDR_OFFSETID", "ROW_NUMBER"]
```

The precise priority should be validated against duplicate timestamps and against the exchange feed logic. `SEQUENCETIME` is the economically meaningful synchronization time; the `HDR_*` fields are technical tie-breakers.

---

## 3. Event and order identity

| Column | Source field in document | Native datatype | Meaning |
|---|---:|---:|---|
| `EVENTID` | `eventid` | `varchar2(50 char)` | Event identifier. |
| `ORDEREVENTTYPE (*)` | `ordereventtype` | `number(3)` | Type of order event. The document lists examples: new, modify, fill, cancel, reject, stop triggered, iceberg refill, market-to-limit transformed into limit, VFA, VFC, and collar breach confirmation. |
| `TRADETYPE (*)` | `tradetype` | `number(3)` | Type of trade. The code dictionary is not included in the provided document. |
| `ORDERSTATUS` | likely decoded from `orderqualifiers` | derived indicator | Order status is described inside `orderqualifiers`: bit position for order status indicates inactive versus active, with `0 = inactive` and `1 = active`. The selected column is probably a decoded version of that bit. |
| `ORDERID` | `orderid` | `varchar2(50 char)` | Numerical order identifier assigned by the Matching Engine. It is unique per instrument and Exchange Market Mechanism. |
| `ORDERPRIORITY` | `orderpriority` | `varchar2(50 char)` | Priority rank of the order. The lowest value has the highest priority. The document states that it is unique per symbol index and Exchange Market Mechanism and is also used as the unique order identifier in the market-data feed. |
| `CLIENTORDERID` | `clientorderid` | `varchar2(50 char)` | Identifier assigned by the client when submitting an order to the exchange. It is returned in outbound messages so that clients can reconcile responses with the original inbound request. |
| `ORIGCLIENTORDERID` | `origclientorderid` | `varchar2(50 char)` | Client order ID of the original order. Used to identify the previous order on cancel and replacement requests. |
| `EXECUTIONID` | `executionid` | `number(10)` | Execution identifier, unique per instrument and per day. It identifies a trade per instrument and is populated for fill, partial fill, and trade-cancellation events. The same execution ID is reported on both sides of the trade and reused for trade bust notifications. |
| `TRADEUNIQUEIDENTIFIER` | `tradeuniqueidentifier` | `varchar2(16 char)` | Trade Unique Identifier, also described as the trading venue transaction identification code. It is unique, consistent, and persistent per ISO 10383 segment MIC and per trading day. |

---

## 4. Order-book and trade mechanics

| Column | Source field in document | Native datatype | Meaning |
|---|---:|---:|---|
| `ORDERSIDE (*)` | `orderside` | `number(3)` | Side of the order. The document notes that the value `Cross` is used only for order entry and is never populated in the market-data feed. The numeric side mapping is not included in the provided document. |
| `ORDERPX` | `orderpx` | `number(30,8)` | Instrument price per quantity unit. The document says this must be calculated with Price/Index Level Decimals. It is null for priceless orders in market data; in order entry it is mandatory for priced orders such as limit and stop-limit orders and null when price is irrelevant, such as market, stop-market, peg, or market-to-limit orders. |
| `ORDERQTY` | `orderqty` | `number(30,8)` | Total order quantity per quantity unit. The document says this must be calculated with Quantity Decimals. |
| `DISPLAYEDQTY` | `displayedqty` | `number(30,8)` | Quantity displayed to the market. The document explicitly associates this field with iceberg orders. |
| `LEAVESQTY` | `leavesqty` | `number(30,8)` | Remaining quantity of an order, i.e. the quantity still open for further execution. |
| `LASTSHARES` | `lastshares` | `number(30,8)` | Last traded quantity: the quantity of the last fill on the instrument. The document says this must be calculated with Quantity Decimals. |
| `LASTTRADEDPX` | `lasttradedpx` | `number(30,8)` | Last traded price: the price of the last fill on the instrument. The document says this must be calculated with Price/Index Decimals. |
| `ORDERTYPE (*)` | `ordertype` | `number(3)` | Type of order. The document notes that stop-market, stop-limit, average-price, iceberg, and mid-point peg values are used only for order entry and are never populated in the market-data feed. The numeric code dictionary is not included in the provided document. |
| `TIMEINFORCE (*)` | `timeinforce` | `number(3)` | Maximum validity of the order. For stop orders, it gives the maximum validity while the stop order is not yet triggered. The code dictionary is not included in the provided document. |
| `KILLREASON (*)` | `killreason` | `number(5)` | Order kill reason. The source document gives the semantic role but not the code dictionary. |
| `PASSIVEORDER` | likely decoded from `tradequalifier` or `orderqualifiers` | derived indicator | The document defines a passive-order bit in `tradequalifier`: bit position 2 indicates whether the corresponding order was passive, with `0 = no`, `1 = yes`. At the order level, `orderqualifiers` also defines an aggressive-order bit with `0 = passive order`, `1 = aggressive order`. You should verify which field your ETL used to derive this column. |
| `AGGRESSIVEORDER` | likely decoded from `tradequalifier` or `orderqualifiers` | derived indicator | The document defines an aggressive-order bit in `tradequalifier`: bit position 3 indicates whether the corresponding order was aggressive, with `0 = no`, `1 = yes`. At the order level, `orderqualifiers` also defines an aggressive-order bit with `0 = passive order`, `1 = aggressive order`. You should verify which field your ETL used to derive this column. |

### Passive/aggressive interpretation

The document contains two relevant bit-field descriptions:

- `tradequalifier`: trade-level bits include `Passive Order` and `Aggressive Order`.
- `orderqualifiers`: order-level bits include `Order Status` and `Aggressive Order`.

This distinction matters. For a fill event, passive/aggressive status should normally refer to the role played by the order in the trade. For a standing order-book event, an order-level aggressiveness flag may instead describe whether the order was marketable/aggressive upon entry. Before using `PASSIVEORDER` and `AGGRESSIVEORDER` in empirical work, check whether they were decoded from `tradequalifier` or `orderqualifiers`.

---

## 5. Agent, client, and regulatory identity

| Column | Source field in document | Native datatype | Meaning |
|---|---:|---:|---|
| `NMSC_ORIGINALCLIENTIDSHORTCODE` | `nmsc_originalclientidshortcode` | `number(10)` | Original MiFID II client-identification short code. It is the first client-identification short code associated with a new order and remains unchanged over the whole order life. |
| `MSC_EVENTCLIENTIDSHORTCODE` | `msc_eventclientidshortcode` | `number(10)` | Event MiFID II client-identification short code. It is associated with order events except for a new order and can change during the order life. |
| `FIRMID` | `firmid` | `varchar2(8 char)` | Identifier of the member firm sending the message. The document says it is provided by the exchange upon firm registration. |
| `NMSC_ORIGINALEXECWFIRMSHORTCODE` | `nmsc_originalexecwfirmshortcode` | `number(10)` | Original MiFID II short code for execution within firm. It identifies the trader or algorithm responsible for execution making. It is the first such short code associated with a new order and remains unchanged over the order life. |
| `MSC_EVENTEXECWFIRMSHORTCODE` | `msc_eventexecwfirmshortcode` | `number(10)` | Event MiFID II short code for execution within firm. It identifies the trader or algorithm responsible for execution making at the event level and can change during the order life. The document text appears to contain a copy-paste inconsistency saying “Client Identification Short Code”, but the field name and ESMA description refer to execution within firm. |
| `NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE` | `nmsc_originalinvestdeciswfirmshortcode` | `number(10)` | Original MiFID II short code for investment decision within firm. It identifies the trader or algorithm responsible for the investment decision and remains unchanged over the order life. |
| `NMSC_ORIGINALNONEXECBROKERSHORTCODE` | `nmsc_originalnonexecbrokershortcode` | `number(10)` | Original MiFID II short code for the non-executing broker. It remains unchanged over the order life and should be blank when not relevant. |
| `ACCOUNTTYPEINTERNAL (*)` | `accounttypeinternal` | `number(3)` | Internal account-type field managed by IACA Finish. It indicates the account type for which the order is entered, for example client account, house account, or liquidity-provider account. For cross orders, it refers to the buy side of the cross order. The document states that non-LP clients cannot use type `6` for liquidity provider and that only Retail Member Organizations can send type `3` retail orders on behalf of retail clients. |
| `LPROLE (*)` | `lprole` | `number(3)` | Liquidity-provider role. It identifies the type of liquidity provider when `accounttypeinternal` equals liquidity provider. |
| `ORDER_TRADINGCAPACITY (*)` | `nmof_tradingcapacity` | `number(3)` | Trading capacity of the order submission: matched principal, own account, or another capacity. The selected column appears renamed from the source field `nmof_tradingcapacity`. |
| `DEAINDICATOR` | decoded from `mifidindicators` | derived indicator | Direct Electronic Access indicator. The document defines it inside `mifidindicators`: `0 = no`, `1 = yes`. If set to `1`, the client-identification short code must be populated. |
| `INVESTMENTALGOINDICATOR` | decoded from `mifidindicators` | derived indicator | Investment algorithm indicator. The document defines it inside `mifidindicators`: `0 = no algorithm involved`, `1 = algorithm involved`. If set to `1`, the investment-decision-within-firm short code must be filled. |
| `EXECUTIONALGOINDICATOR` | decoded from `mifidindicators` | derived indicator | Execution algorithm indicator. The document defines it inside `mifidindicators`: `0 = no algorithm involved`, `1 = algorithm involved`. |

---

## 6. Native fields versus renamed or decoded fields

The following selected columns do **not** appear with exactly the same name in the source document and should be audited against the ETL logic.

| Selected column | Likely source | Type of transformation | Check to perform |
|---|---|---|---|
| `ISIN` | `codisin` | Rename | Confirm that `ISIN == codisin`. |
| `ROW_NUMBER` | none in source document | ETL-derived row index | Confirm whether it is deterministic within `(TRADEDATE, ISIN)` or over the whole extract. |
| `ORDERSTATUS` | `orderqualifiers` | Bit-field decoding | Confirm bit position and mapping. The document gives `0 = inactive`, `1 = active`. |
| `PASSIVEORDER` | `tradequalifier` or `orderqualifiers` | Bit-field decoding | Confirm whether it is trade-level or order-level. |
| `AGGRESSIVEORDER` | `tradequalifier` or `orderqualifiers` | Bit-field decoding | Confirm whether it is trade-level or order-level. |
| `ORDER_TRADINGCAPACITY (*)` | `nmof_tradingcapacity` | Rename | Confirm that the values are copied from `nmof_tradingcapacity`. |
| `DEAINDICATOR` | `mifidindicators` | Bit-field decoding | Confirm bit position and Boolean mapping. |
| `INVESTMENTALGOINDICATOR` | `mifidindicators` | Bit-field decoding | Confirm bit position and Boolean mapping. |
| `EXECUTIONALGOINDICATOR` | `mifidindicators` | Bit-field decoding | Confirm bit position and Boolean mapping. |

---

## 7. Suggested interpretation for empirical market-microstructure work

### Order life-cycle reconstruction

Use these fields to reconstruct order histories:

- `ORDERID`: exchange/matching-engine order identity.
- `CLIENTORDERID`: client-supplied identifier for the inbound message.
- `ORIGCLIENTORDERID`: previous client order ID for cancel/replace workflows.
- `ORDEREVENTTYPE (*)`: event type along the order life cycle.
- `ORDERSTATUS`, `LEAVESQTY`, `KILLREASON (*)`: state and termination information.

A typical event-level order-state reconstruction would sort by instrument and event time, then update state variables such as outstanding quantity, displayed quantity, price, side, and status.

### Trade reconstruction

Use these fields to identify executions:

- `EXECUTIONID`: unique per instrument and day; appears on both sides of a trade.
- `TRADEUNIQUEIDENTIFIER`: persistent venue transaction identifier.
- `LASTSHARES`, `LASTTRADEDPX`, `TRADETIME`: executed quantity, price, and timestamp.
- `PASSIVEORDER`, `AGGRESSIVEORDER`: role of the order in the trade, subject to the caveat above.

### Agent and client classification

Use these fields to study participant behavior:

- `FIRMID`: member firm submitting the message.
- `NMSC_ORIGINALCLIENTIDSHORTCODE`: stable client short code over the order life.
- `MSC_EVENTCLIENTIDSHORTCODE`: event-level client short code, potentially changing over the order life.
- `NMSC_ORIGINALEXECWFIRMSHORTCODE` and `MSC_EVENTEXECWFIRMSHORTCODE`: trader/algorithm responsible for execution.
- `NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE`: trader/algorithm responsible for investment decision.
- `ACCOUNTTYPEINTERNAL (*)`, `LPROLE (*)`, `ORDER_TRADINGCAPACITY (*)`: account role, liquidity-provider role, and trading capacity.
- `DEAINDICATOR`, `INVESTMENTALGOINDICATOR`, `EXECUTIONALGOINDICATOR`: regulatory flags for direct electronic access and algorithmic decision/execution involvement.

For behavioural studies, the distinction between **original** short codes and **event** short codes is important: original short codes are stable over the order life, while event short codes can change over modifications and other events.

---

## 8. Minimal validation checks

Before using the filtered data, the following checks are recommended.

```python
# 1. Check that renamed fields are really present under the expected names.
missing = [c for c in keep_cols if c not in df.columns]
print(missing)

# 2. Check duplicate event-ordering keys.
ordering_key = [
    "TRADEDATE", "ISIN", "SEQUENCETIME",
    "HDR_APPLKEYSEQUENCENUMBER", "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID", "ROW_NUMBER",
]

# Polars example:
df.group_by(ordering_key).len().filter(pl.col("len") > 1)

# 3. Check whether prices and quantities look already scaled.
df.select([
    pl.col("ORDERPX").min().alias("min_order_px"),
    pl.col("ORDERPX").max().alias("max_order_px"),
    pl.col("ORDERQTY").min().alias("min_order_qty"),
    pl.col("ORDERQTY").max().alias("max_order_qty"),
])

# 4. Check derived Boolean columns.
for col in ["ORDERSTATUS", "PASSIVEORDER", "AGGRESSIVEORDER",
            "DEAINDICATOR", "INVESTMENTALGOINDICATOR", "EXECUTIONALGOINDICATOR"]:
    print(col, df.select(pl.col(col).unique().sort()))
```

---

## 9. Summary table

| Conceptual block | Main columns | Main use |
|---|---|---|
| Instrument/session | `TRADEDATE`, `ISIN` | Partition the data by trading day and instrument. |
| Ordering/timestamps | `SEQUENCETIME`, `BOOKIN`, `BOOKOUTTIME`, `TRADETIME`, `HDR_*`, `ROW_NUMBER` | Reconstruct the event sequence and resolve ordering ties. |
| Order identity | `ORDERID`, `ORDERPRIORITY`, `CLIENTORDERID`, `ORIGCLIENTORDERID` | Link events belonging to the same order and reconcile client/exchange identifiers. |
| Trade identity | `EXECUTIONID`, `TRADEUNIQUEIDENTIFIER`, `TRADETYPE (*)` | Identify executions and trade-level events. |
| Mechanics | `ORDERSIDE (*)`, `ORDERPX`, `ORDERQTY`, `DISPLAYEDQTY`, `LEAVESQTY`, `LASTSHARES`, `LASTTRADEDPX`, `ORDERTYPE (*)`, `TIMEINFORCE (*)`, `KILLREASON (*)` | Reconstruct order-book dynamics and executed trades. |
| Aggressor/passive role | `PASSIVEORDER`, `AGGRESSIVEORDER` | Classify the role of an order or trade side, depending on ETL decoding. |
| Participant identity | `FIRMID`, `NMSC_*`, `MSC_*` | Identify member, client, trader, broker, and algorithm short codes. |
| Regulatory / account role | `ACCOUNTTYPEINTERNAL (*)`, `LPROLE (*)`, `ORDER_TRADINGCAPACITY (*)`, `DEAINDICATOR`, `INVESTMENTALGOINDICATOR`, `EXECUTIONALGOINDICATOR` | Classify account type, liquidity-provider status, trading capacity, DEA access, and algorithmic involvement. |
