"""bioRxiv search tool for literature mining."""
import os
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta


def _days_back_from_env(default_days: int = 730) -> int:
    raw = os.getenv("BIORXIV_DAYS_BACK")
    if not raw:
        return default_days
    try:
        value = int(raw)
        return max(1, value)
    except ValueError:
        return default_days


def biorxiv_search_func(
    query: str,
    max_results: int = 50,
    anchor: str = None,
    days_back: Optional[int] = None,
) -> List[Dict]:
    """
    Searches bioRxiv for preprints in BOTH title and abstract.
    Filters results to ensure anchor entity is present.
    """
    try:
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 50
        # bioRxiv API is date-based, so we search recent papers
        end_date = datetime.now()
        window_days = days_back if days_back is not None else _days_back_from_env()
        start_date = end_date - timedelta(days=window_days)
        
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
        query_terms = [t.strip() for t in query_clean.split() if len(t.strip()) > 2]
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            combined = title + ' ' + abstract
            
            # If anchor provided, it MUST be present
            if anchor:
                anchor_lower = anchor.lower()
                if anchor_lower not in combined:
                    continue
            
            # Check if query terms match (relaxed to avoid empty results)
            matches = sum(1 for term in query_terms if term in combined)
            min_matches = 0 if not query_terms else max(1, len(query_terms) // 3)
            
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
