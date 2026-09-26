"""Display-only redaction for proprietary identifiers.

I wrote this so I can show data samples in notebooks without leaking real
customer, transaction, device or IP identifiers into saved outputs (and from
there into git/GitHub). The golden rule of this file: it only ever touches a
COPY made for display. My exploration frames, CSVs and parquet files are never
modified — I use the real frame for analysis and `show()` for the shop window.

Usage inside a notebook (run from notebooks/, like the rest of this project):
    import sys; sys.path.insert(0, "..")
    from redact import show
    show(df_raw)        # redacted sample, safe to save and push
    df_raw.head()       # real data, for my eyes only — never save this output
"""

import pandas as pd

# Columns I treat as identifiers. If one isn't in the frame I skip it quietly,
# so the same call works on raw, cleaned and engineered frames alike.
ID_COLUMNS = ("transaction_id", "customer_id", "device_id", "ip_address")

# Short label per identifier, so the redacted table still reads naturally.
_PREFIX = {"transaction_id": "txn", "customer_id": "cust",
           "device_id": "dev", "ip_address": "ip"}


def redact_frame(frame, n=50):
    """Return the first n rows with every ID column replaced by stable labels.

    Stable means: the same real ID always gets the same label (cust_001…), so
    a customer appearing twice still lines up in the displayed table — I just
    can't trace them back to anyone. Missing values stay missing. The input
    frame is returned untouched; I work on a copy.
    """
    view = frame.head(n).copy()
    for col in ID_COLUMNS:
        if col not in view.columns:
            continue
        # factorize maps each distinct value to 0,1,2… (missing -> -1),
        # which is exactly the numbering I need.
        codes, _ = pd.factorize(view[col].astype(object))
        prefix = _PREFIX.get(col, "id")
        view[col] = pd.Series(
            [f"{prefix}_{c + 1:03d}" if c >= 0 else float("nan") for c in codes],
            index=view.index,
        )
    return view


def show(frame, n=50):
    """Display-safe head(): a redacted sample, original untouched.

    I call this instead of df.head() whenever the output will be saved,
    screenshotted or pushed anywhere. Exploration keeps using the real frame.
    """
    return redact_frame(frame, n)
