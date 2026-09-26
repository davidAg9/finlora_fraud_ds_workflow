# My EDA & Data Quality Log — FinLora Fraud Detection

I worked through `FinLora_Customer_Transaction_Dataset.csv` step by step so I can
defend every cleaning decision later. This is my working log, written the way I'd
explain it to another person, with the numbers kept exact.

Tools I used: Python 3.13, pandas 3, numpy, scikit-learn (pixi env, conda-forge).

---

## What I started with

| What I checked | What I found |
|---|---|
| Raw shape | 11,400 rows × 26 columns |
| Fraud rate | 8.75% (997 / 11,400) — quite imbalanced |
| Unique customers | 1,315 (the top 5 IDs hold 54% of all transactions; busiest has 1,510) |
| Unique devices | 2,113 |
| Timestamp range | 2022-10-03 → 2025-12-16 (UTC) |

---

## Step 1 — I audited the shape of the data and where values were missing

I ran `df.info()` and a missing-value table (notebook cells 3–4). Seven columns had gaps:

| Column | Missing | % |
|---|---|---|
| amount_usd | 305 | 2.68% |
| ip_address | 305 | 2.68% |
| ip_country | 301 | 2.64% |
| kyc_tier | 300 | 2.63% |
| fee | 295 | 2.59% |
| device_trust_score | 295 | 2.59% |
| timestamp | 29 | 0.25% |

**What I noticed:** the ~300-row gaps overlap almost perfectly — a row missing
`amount_usd` is usually also missing `ip_address`, `ip_country`, `kyc_tier`, `fee`
and `device_trust_score`. So this is one correlated outage, not random gaps, which
means the missingness itself might carry information.

I checked fraud rate by how many fields a row is missing:

| # missing fields | rows | fraud rate |
|---|---|---|
| 0 | 11,066 | 8.93% |
| 1 (timestamp only) | 29 | 3.45% |
| 4–5 | 15 | 0.00% |
| 6 (the block) | 290 | 2.76% |

Incomplete records are *less* likely to be fraud here, not more. My decision: **don't
silently drop them** — I added a `record_incomplete` flag and imputed the values
separately so the model can learn that pattern.

---

## Step 2 — I checked whether repeated transaction IDs were real duplicates

Before dropping anything I wanted proof. What I found:

1. 400 rows share an ID with at least one other row
2. They form exactly 200 pairs — no triplets
3. Inside each pair, **every column is identical** (I checked with a groupby)
4. A whole-row duplicate check agreed exactly: 200 duplicates
5. No pair disagreed on the fraud label — dedup keeps every distinct transaction,
   fraud count only goes 997 → 995 because 2 fraud rows were stored twice

**My decision:** drop the exact duplicates. Duplicates inflate metrics and can leak
the same transaction into both train and test, so this had to go first.

---

## Step 3 — I fixed the `amount_src` column, which wouldn't parse as numbers

Parsing failed on values like `9,998.85` — four rows had thousands separators
(commas), which forced the whole column to load as text. My fix: strip the commas,
then parse, coercing anything left over to missing:

```python
df['amount_src'] = pd.to_numeric(
    df['amount_src'].astype('string').str.replace(',', '', regex=False),
    errors='coerce',
)
```

I also found 103 rows where `amount_src` was negative but `amount_usd` was positive
(a flipped sign upstream), so I took the absolute value after parsing. Sanity check
afterwards: `amount_usd / amount_src` centers exactly on 1.0, which is what I'd expect.

---

## Step 4 — I masked placeholder numbers that were pretending to be measurements

Some values are sentinels, not real readings:

| Column | Bad value | Rows | What I did |
|---|---|---|---|
| fee | -1 | 101 | masked to missing (real fees sit around 3.50) |
| fee | 9999.99 | 103 | masked to missing (obvious placeholder) |
| txn_velocity_1h | -1 | 204 | masked to missing (counts can't be negative) |
| ip_risk_score | > 1.0 | 204 | clipped to 1.0 (the valid range is 0–1) |
| device_trust_score | -0.1 | 204 | clipped to 0.0 (the valid range is 0–1) |

The three 204-row problems land on the **same rows** — a second corrupted batch,
mostly legitimate customers (2.0% fraud). Masking to missing lets one imputation
path handle them, and stops tree models from splitting on `fee == 9999.99` as if it
meant something.

---

## Step 5 — I cleaned up messy category labels

`channel` and `kyc_tier` had case variants and typos: `ATM/ATm/atm`,
`MOBILE/mobile/mobille`, `WEB/web/weeb`, `STANDARD/standrd/standard`,
`ENHANCED/enhancd/enhanced`, plus the literal text `"NAN"` in `kyc_tier` (a string,
invisible to missing-value checks — I caught it with a value-counts table).

One pattern jumped out: raw `kyc_tier == 'low'` had a **51.6%** fraud rate vs ~2%
elsewhere. That's suspiciously strong, so I flagged it for a leakage review during
the SHAP analysis (spoiler: after cleaning it settles at ~51.2% and stays the
strongest category signal, driven by genuinely risky low-KYC accounts).

My fix: strip whitespace, lowercase everything, map the typos, turn `"nan"` into a
real missing value.

---

## Step 6 — I dealt with broken timestamps

61 values were unusable: 29 missing, 21 × `"0000-00-00T00:00:00Z"`, 11 × impossible
dates like `"2025/13/40 25:61:00"`. Fraud rate in those rows is only 3.3%. I coerced
the bad ones to missing and kept the rows rather than throwing away 61 transactions
over a timestamp.

---

## Step 7 — A few customers dominate, so I chose the split carefully

The top 5 customers hold 54.3% of all rows. A random row-wise split would put the
same customer's transactions in both train and test, giving optimistic scores that
wouldn't survive deployment. So:

- My main evaluation is a **time-based split** (train on the past, test on the
  future), which mirrors how the model would actually be used.
- I also sanity-check with a customer-grouped split.
- Any per-customer behaviour feature may only use *prior* history — never a
  full-history average (that would leak the future into training).

---

## The amount finding I almost missed (my favourite part)

My early charts showed `amount_usd` as skewed and I nearly moved on. Then I asked a
better question: *is there an amount band where fraud concentrates?* Yes — and it's
not "bigger amount = more fraud":

| Amount band | Transactions | Frauds | Fraud rate |
|---|---|---|---|
| 0–50 | 851 | 18 | 2.1% |
| 50–100 | 2,285 | 41 | 1.8% |
| 100–250 | 4,513 | 168 | 3.7% |
| 250–500 | 2,128 | 234 | 11.0% |
| 500–1k | 840 | 240 | 28.7% |
| 1k–2.5k | 307 | 215 | **70.0%** |
| 2.5k–5k | 49 | 47 | **95.9%** |
| 5k–10k | 192 | 26 | 13.5% |
| 10k+ | 32 | 5 | 15.6% |

Fraud has a sweet spot around $250–$5k and then *falls off a cliff* above $5k. I
tested candidate cut-offs as standalone flags:

| Cut-off | Share of all fraud caught | Fraud rate above cut-off | Standalone AUC |
|---|---|---|---|
| ≥ $250 | 77.2% | 21.6% | **0.75** |
| ≥ $500 | 53.7% | 37.6% | 0.72 |
| ≥ $1,000 | 29.4% | 50.5% | 0.63 |

I picked **$250**: it catches more than three-quarters of all fraud at ~2.4× the
baseline rate, and it's a clean number I can explain to anyone. That's the
`amount_high_flag` feature I added to the model. (For reference, plain
`log1p(amount)` scores 0.82 on its own — the flag adds the crisp threshold the
smooth curve can't express.)

---

---

## Follow-up: I tested the diagram's Step-2 and Step-4 features (and dropped half)

A reference design I compared against suggested time features (hour, day-of-week,
weekend, late-night) and risk flags (high IP risk, low device trust, new/very new
account, velocity spike). I don't add features on authority — I tested each one:

| Candidate | Fraud signal I measured | Standalone AUC | Kept? |
|---|---|---|---|
| hour of day (4–8h spike: 18–21% vs 8.9%) | real but weak | 0.61 | No |
| late-night (0–5h: 11.5% vs 8.0%) | tiny | 0.54 | No |
| day of week / weekend | none (8.3–9.8% everywhere) | — | No |
| high_ip_risk (≥0.7: 58% fraud) | strong | 0.85 | **Yes** |
| low_device_trust (≤0.3: 85% fraud) | very strong | 0.78 | **Yes** |
| new_account (<90d: 44% fraud) | strong | 0.85 | **Yes** |
| very_new_account (<30d: 36% fraud) | strong | 0.77 | **Yes** |
| velocity_spike (≥3/hr: ~79% fraud) | strong | 0.87 | **Yes** |

Retraining with all eight: PR-AUC 0.9632 → 0.9653. With only the five flags:
0.9652 — so the time features contributed nothing and I dropped them (this also
keeps `timestamp` out of the serving contract, which is one less thing the API
needs). The five flags stayed: small gain, no extra serving cost, each cut backed
by the table above. Full story in my modelling log.

One thing I want on record: just because hour/weekend carry no signal *today*
doesn't mean they're dead. Fraud runs on human schedules — if attacker behaviour
shifts (say, a crew moves to weekend bursts to dodge analyst coverage), these are
the first features I'd re-test. I keep the hour/weekend checks as a standing
future experiment, not a closed case.

## Where this left me

1. One cleaning routine covering Steps 2–6, reused for serving so training and the
   API never drift.
2. A battery of post-clean assertions (unique IDs, valid ranges, correct types).
3. Target-aware EDA: fraud-vs-legit separation per feature (the charts in my EDA
   notebook).
4. A time-based split set up before any fitted preprocessing touches the data.
5. A new derived feature, `amount_high_flag`, with the evidence table above to
   justify it.
