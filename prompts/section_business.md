Draft the **Business Overview** section of a diligence memo on {{target}}.

Cover, in this order and only where the filings support it:

- What the company sells and to whom — segments, principal products or services, revenue mix
  by segment or geography if disclosed.
- Scale: revenue, employees, geographic footprint.
- How the business model works — where the margin comes from, recurring vs transactional.
- Any material change in the business during the covered period (acquisition, divestiture,
  reorganization, new segment reporting).

Retrieve before you write. `search_filings` filtered to the target's 10-K Item 1 material is
the fastest route; `get_financials` gives exact revenue and segment-level figures where the
XBRL tags exist.

Every claim carries an evidence id. Where the filings do not disclose something on this list,
omit it silently rather than hedging — the memo's Open Diligence Questions section handles
gaps.

Call `draft_section` with section `Business Overview` when done.
