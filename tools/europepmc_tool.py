"""Europe PMC search tool - larger database than PubMed."""
import requests
from typing import List, Dict


def europepmc_search_func(query: str, max_results: int = 100, anchor: str = None) -> List[Dict]:
    """
    Search Europe PMC for biomedical literature.
    Europe PMC has broader coverage than PubMed.
    """
    try:
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 100
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            'query': query,
            'format': 'json',
            'pageSize': min(max_results, 1000)  # Max 1000 per request
        }
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            return [{'error': f"Europe PMC HTTP {response.status_code}"}]
        data = response.json()
        
        results = []
        papers = data.get('resultList', {}).get('result', [])
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstractText', '').lower()
            combined = title + ' ' + abstract
            
            # If anchor provided, check if it's present
            if anchor:
                anchor_lower = anchor.lower()
                if anchor_lower not in combined:
                    continue
            
            # Extract authors
            authors = []
            author_list = paper.get('authorList', {}).get('author', [])
            for author in author_list[:5]:
                full_name = f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
                if full_name:
                    authors.append(full_name)
            
            results.append({
                'title': paper.get('title', ''),
                'authors': authors,
                'abstract': paper.get('abstractText', 'No abstract available'),
                'published': paper.get('firstPublicationDate', ''),
                'pmid': paper.get('pmid', ''),
                'pmcid': paper.get('pmcid', ''),
                'doi': paper.get('doi', ''),
                'source': 'Europe PMC',
                'journal': paper.get('journalTitle', ''),
                'open_access': paper.get('isOpenAccess', 'N') == 'Y'
            })
            
            if len(results) >= max_results:
                break
        
        return results
    
    except Exception as e:
        return [{'error': f"Europe PMC search failed: {str(e)}"}]


class EuropePMCSearchTool:
    def __init__(self):
        self.name = "Europe PMC Search"
        self.description = "Searches Europe PMC for biomedical literature"
    
    def _run(self, query: str, max_results: int = 100, anchor: str = None) -> List[Dict]:
        return europepmc_search_func(query, max_results, anchor)
