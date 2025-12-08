# BioAgentHub Crawler (from PEproject)

Lightweight copy of the PEproject literature crawler to fetch open-access papers for any topic.
- Sources: Europe PMC + bioRxiv (open access).
- Downloads PDFs with multiple fallbacks (PMC, Europe PMC render, Unpaywall, DOI redirects).
- No LLM needed; only `requests` required.

## Quickstart
```bash
cd UIUC/work/BioAgentHub
python crawler/simple_crawl.py --query "PETase depolymerase" --max 5 --download 1 --out crawler_outputs
```
Outputs are stored under `crawler_outputs/run_<timestamp>/`:
- `papers.json` — merged & deduplicated metadata.
- `download_log.json` — status for attempted PDF downloads.
- PDFs for any successful downloads.

## Notes
- Code copied from `/Users/viku/Documents/UIUC/work/PEproject` (query expansion + download stack).
- You can change `--anchor` to require a term in title/abstract (e.g., enzyme name).
- Increase `--download` to pull more PDFs (respect rate limits).
