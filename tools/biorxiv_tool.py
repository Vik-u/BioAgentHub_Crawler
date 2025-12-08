"""bioRxiv search tool for literature mining."""
import requests
from typing import List, Dict
from datetime import datetime, timedelta


def biorxiv_search_func(query: str, max_results: int = 50, anchor: str = None) -> List[Dict]:
    """
    Searches bioRxiv for preprints in BOTH title and abstract.
    Filters results to ensure anchor entity is present.
    """
    try:
        # bioRxiv API is date-based, so we search recent papers
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # Last 2 years for better coverage
        
        base_url = "https://api.biorxiv.org/details/biorxiv"
        url = f"{base_url}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}/0"
        
        response = requests.get(url, timeout=30)
        data = response.json()
        
        results = []
        papers = data.get('collection', [])
        
        # Extract query terms for matching
        query_lower = query.lower()
        # Remove boolean operators for term extraction
        query_clean = query_lower.replace(' and ', ' ').replace(' or ', ' ').replace('"', '')
        query_terms = [t.strip() for t in query_clean.split() if len(t.strip()) > 3]
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            combined = title + ' ' + abstract
            
            # If anchor provided, it MUST be present
            if anchor:
                anchor_lower = anchor.lower()
                if anchor_lower not in combined:
                    continue
            
            # Check if query terms match (at least 2 terms or 30% of terms)
            matches = sum(1 for term in query_terms if term in combined)
            min_matches = max(2, len(query_terms) // 3)
            
            if matches >= min_matches:
                results.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', '').split('; '),
                    'abstract': paper.get('abstract', ''),
                    'published': paper.get('date', ''),
                    'doi': paper.get('doi', ''),
                    'source': 'bioRxiv',
                    'open_access': True,  # bioRxiv is always open access
                    'category': paper.get('category', ''),
                    'relevance_score': matches  # For sorting
                })
                
                if len(results) >= max_results * 2:  # Fetch extra for sorting
                    break
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return results[:max_results]
    except Exception as e:
        return [{'error': f"bioRxiv search failed: {str(e)}"}]


# Wrapper class for compatibility
class BioRxivSearchTool:
    def __init__(self):
        self.name = "bioRxiv Search"
        self.description = "Searches bioRxiv for biology preprints"
    
    def _run(self, query: str, max_results: int = 50, anchor: str = None) -> List[Dict]:
        return biorxiv_search_func(query, max_results, anchor)
