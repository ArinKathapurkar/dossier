Draft the **Competitive Position** section of a diligence memo on {{target}}.

The deal's peer set is: {{peers}}.

Cover:

- How the company describes its own competitive position and the basis on which it competes,
  in its own words from the filings.
- Named competitors, customers and suppliers. `graph_query` is the right tool here — it
  crosses documents, which repeated `search_filings` calls will not do efficiently.
- A quantitative comparison against the peers on whatever concepts the XBRL facts support
  for a shared fiscal year — use `compare_peers`, and say which year you used.
- Concentration: a customer, supplier or geography the filings flag as material.

Do not assert market share unless a filing states one. Do not rank the company against peers
on anything except figures you retrieved.

Call `draft_section` with section `Competitive Position` when done.
