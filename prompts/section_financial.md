Draft the **Financial Profile** section of a diligence memo on {{target}}.

Build a picture of the financial shape of the business over the periods available:

- Revenue and growth. Use `get_financials` for the figures and `compute` with `pct_change` or
  `cagr` for growth rates — never compute a rate in your head.
- Profitability: operating income and net income, and the margins implied by them.
- Cash generation: operating cash flow, capital expenditure, and free cash flow as the
  difference between them.
- Balance-sheet position: assets, liabilities, equity, cash, long-term debt. Leverage where
  the components are available.

Present the numbers in a table with the fiscal year, the figure, and its evidence id. Every
derived figure gets its own `[C#]` id from `compute`, so the arithmetic is checkable.

Where a concept is missing from the XBRL facts for a period, say the period is unavailable
rather than substituting a nearby year.

Call `draft_section` with section `Financial Profile` when done.
