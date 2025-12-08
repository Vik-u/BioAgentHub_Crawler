#!/usr/bin/env python3
"""
IMPROVED PDF Downloader - Multiple strategies for open access papers.
"""
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class ImprovedPDFDownloader:
    """Better PDF download with multiple fallback strategies."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_pmcid(self, pmid):
        """Convert PMID to PMCID."""
        if not pmid:
            return None
        try:
            # Convert to string and clean
            pmid_str = str(pmid).strip()
            url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid_str}&format=json"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                records = data.get('records', [])
                if records and len(records) > 0:
                    pmcid = records[0].get('pmcid')
                    if pmcid:
                        return pmcid
        except Exception as e:
            # Debug: print error
            print(f"      PMCID lookup error for {pmid}: {e}")
        return None
    
    def try_pmc_oa_pdf(self, pmcid):
        """Try PMC Open Access PDF service."""
        if not pmcid:
            return None
        try:
            pmcid_clean = str(pmcid).replace('PMC', '')
            # Try OA PDF service
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_clean}/pdf/"
            r = self.session.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                return url
        except:
            pass
        return None
    
    def try_europepmc_pdf(self, pmcid):
        """Try Europe PMC PDF - just return URL, don't test."""
        if not pmcid:
            return None
        try:
            pmcid_clean = str(pmcid).replace('PMC', '')
            url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmcid_clean}&blobtype=pdf"
            return url  # Return directly, test during download
        except:
            pass
        return None
    
    def try_unpaywall(self, doi):
        """Try Unpaywall API."""
        if not doi:
            return None
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get('is_oa'):
                    # Try best location
                    if data.get('best_oa_location'):
                        pdf_url = data['best_oa_location'].get('url_for_pdf')
                        if pdf_url:
                            return pdf_url
                    # Try any location
                    for loc in data.get('oa_locations', []):
                        pdf_url = loc.get('url_for_pdf')
                        if pdf_url:
                            return pdf_url
        except:
            pass
        return None
    
    def try_doi_org_redirect(self, doi):
        """Follow DOI redirect and look for PDF link."""
        if not doi:
            return None
        try:
            url = f"https://doi.org/{doi}"
            r = self.session.get(url, timeout=10, allow_redirects=True)
            
            # Check if final URL has PDF
            final_url = r.url
            
            # Common OA publisher patterns
            pdf_patterns = [
                # BMC/Springer
                (lambda u: 'biomedcentral.com' in u or 'springeropen.com' in u,
                 lambda u: u.replace('/articles/', '/track/pdf/').replace('.pdf', '') + '.pdf'),
                # PLOS
                (lambda u: 'plos.org' in u,
                 lambda u: u.replace('article?id=', 'article/file?id=') + '&type=printable'),
                # Frontiers
                (lambda u: 'frontiersin.org' in u,
                 lambda u: u + '/pdf'),
                # MDPI
                (lambda u: 'mdpi.com' in u,
                 lambda u: u + '/pdf'),
            ]
            
            for check, transform in pdf_patterns:
                if check(final_url):
                    pdf_url = transform(final_url)
                    # Verify it works
                    test = self.session.head(pdf_url, timeout=5)
                    if test.status_code == 200:
                        return pdf_url
        except:
            pass
        return None
    
    def download_pdf(self, url, output_path):
        """Download PDF from URL."""
        try:
            r = self.session.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify it's a real PDF
                if output_path.stat().st_size > 5000:  # At least 5KB
                    with open(output_path, 'rb') as f:
                        header = f.read(4)
                        if header == b'%PDF':
                            return True
                
                output_path.unlink()  # Delete if not valid
        except:
            pass
        return False
    
    def download_paper(self, paper, output_dir, index):
        """Try all methods to download a paper."""
        title = (paper.get('title') or 'untitled')[:100]
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        
        pmid = paper.get('pmid', '')
        doi = paper.get('doi', '')
        
        filename = f"{index:04d}_{pmid}_{safe_title[:50]}.pdf"
        output_path = Path(output_dir) / filename
        
        if output_path.exists():
            return {'index': index, 'status': 'exists', 'filename': filename}
        
        # Get PMCID if we have PMID
        pmcid = paper.get('pmcid')
        if not pmcid and pmid:
            pmcid = self.get_pmcid(pmid)
        
        # Try all methods in order
        methods = [
            ('Europe PMC', lambda: self.try_europepmc_pdf(pmcid)),
            ('PMC OA', lambda: self.try_pmc_oa_pdf(pmcid)),
            ('Unpaywall', lambda: self.try_unpaywall(doi)),
            ('DOI Redirect', lambda: self.try_doi_org_redirect(doi)),
        ]
        
        for method_name, get_url in methods:
            try:
                pdf_url = get_url()
                if pdf_url:
                    success = self.download_pdf(pdf_url, output_path)
                    if success:
                        return {
                            'index': index,
                            'status': 'success',
                            'method': method_name,
                            'filename': filename,
                            'size': output_path.stat().st_size
                        }
            except:
                continue
        
        return {'index': index, 'status': 'failed', 'title': title[:60]}


def download_batch(results_json, output_dir, max_workers=5):
    """Download PDFs for a batch of papers."""
    
    with open(results_json) as f:
        data = json.load(f)
    
    papers = data['results']
    
    print(f"\n📥 Downloading {len(papers)} papers to {output_dir}")
    print(f"   Using {max_workers} parallel workers\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = ImprovedPDFDownloader()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(downloader.download_paper, p, output_dir, i): i 
                  for i, p in enumerate(papers, 1)}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if len(results) % 10 == 0:
                success = sum(1 for r in results if r['status'] == 'success')
                print(f"  Progress: {len(results)}/{len(papers)} | Downloaded: {success}")
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    exists = sum(1 for r in results if r['status'] == 'exists')
    
    print(f"\n✅ Complete:")
    print(f"   Downloaded: {success}")
    print(f"   Already existed: {exists}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {success/len(papers)*100:.1f}%\n")
    
    # Save log
    with open(output_dir / 'download_log.json', 'w') as f:
        json.dump({'total': len(papers), 'success': success, 'failed': failed, 'details': results}, f, indent=2)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_downloader_v2.py <results.json> [output_dir]")
        sys.exit(1)
    
    results_json = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pdfs"
    
    download_batch(results_json, output_dir)
