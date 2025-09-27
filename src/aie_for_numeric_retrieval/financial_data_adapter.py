#!/usr/bin/env python3
"""
Financial Data Adapter for Existing Text Retrieval System

This adapter converts various existing data formats to AIE framework format
without changing the original data structure.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialDocument:
    """Standardized financial document format for AIE framework"""
    document_id: str
    document_text: str
    query: str
    company: str
    ticker: str
    year: str
    form_type: str
    metadata: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]] = None


class FinancialDataAdapter:
    """Adapter to convert existing financial data formats to AIE framework format"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        
    def load_chunked_documents(self, ticker: str = None, year: str = None, 
                             form_type: str = None, limit: int = None) -> List[FinancialDocument]:
        """Load documents from chunked directory"""
        documents = []
        chunked_path = self.data_root / "chunked"
        
        if not chunked_path.exists():
            logger.warning(f"Chunked directory not found: {chunked_path}")
            return documents
        
        # Get company directories
        company_dirs = [d for d in chunked_path.iterdir() if d.is_dir()]
        if ticker:
            company_dirs = [d for d in company_dirs if d.name == ticker.upper()]
            
        for company_dir in company_dirs:
            ticker_name = company_dir.name
            
            # Get year directories
            year_dirs = [d for d in company_dir.iterdir() if d.is_dir()]
            if year:
                year_dirs = [d for d in year_dirs if d.name == str(year)]
                
            for year_dir in year_dirs:
                year_name = year_dir.name
                
                # Get filing directories
                filing_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
                if form_type:
                    filing_dirs = [d for d in filing_dirs if d.name.startswith(form_type)]
                    
                for filing_dir in filing_dirs:
                    chunks_file = filing_dir / "text_chunks.jsonl"
                    if chunks_file.exists():
                        try:
                            # Parse filing info from directory name
                            form_info = filing_dir.name.split('_')
                            form = form_info[0] if form_info else "Unknown"
                            accno = form_info[1] if len(form_info) > 1 else "Unknown"
                            
                            # Load chunks
                            chunks = self._load_chunks_file(chunks_file)
                            
                            # Combine chunks into document
                            combined_text = "\n\n".join([chunk["content"] for chunk in chunks])
                            
                            # Get metadata from first chunk
                            meta = chunks[0]["meta"] if chunks else {}
                            
                            doc = FinancialDocument(
                                document_id=f"{ticker_name}_{year_name}_{form}_{accno}",
                                document_text=combined_text,
                                query="extract financial information",
                                company=self._get_company_name(ticker_name),
                                ticker=ticker_name,
                                year=year_name,
                                form_type=form,
                                metadata={
                                    "accno": accno,
                                    "source_type": "chunked",
                                    "chunk_count": len(chunks),
                                    "original_meta": meta
                                }
                            )
                            
                            documents.append(doc)
                            
                            if limit and len(documents) >= limit:
                                return documents[:limit]
                                
                        except Exception as e:
                            logger.error(f"Error processing {chunks_file}: {e}")
                            continue
                            
        return documents
    
    def load_processed_text_documents(self, ticker: str = None, year: str = None,
                                    form_type: str = None, limit: int = None) -> List[FinancialDocument]:
        """Load documents from processed text.jsonl files"""
        documents = []
        processed_path = self.data_root / "processed"
        
        if not processed_path.exists():
            logger.warning(f"Processed directory not found: {processed_path}")
            return documents
            
        # Similar structure to chunked
        company_dirs = [d for d in processed_path.iterdir() if d.is_dir()]
        if ticker:
            company_dirs = [d for d in company_dirs if d.name == ticker.upper()]
            
        for company_dir in company_dirs:
            ticker_name = company_dir.name
            
            year_dirs = [d for d in company_dir.iterdir() if d.is_dir()]
            if year:
                year_dirs = [d for d in year_dirs if d.name == str(year)]
                
            for year_dir in year_dirs:
                year_name = year_dir.name
                
                filing_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
                if form_type:
                    filing_dirs = [d for d in filing_dirs if d.name.startswith(form_type)]
                    
                for filing_dir in filing_dirs:
                    text_file = filing_dir / "text.jsonl"
                    if text_file.exists():
                        try:
                            # Parse filing info
                            form_info = filing_dir.name.split('_')
                            form = form_info[0] if form_info else "Unknown"
                            accno = form_info[1] if len(form_info) > 1 else "Unknown"
                            
                            # Load text segments
                            segments = self._load_text_segments(text_file)
                            
                            # Combine segments
                            combined_text = "\n".join([seg["text"] for seg in segments])
                            
                            # Get metadata from first segment
                            meta = segments[0] if segments else {}
                            
                            doc = FinancialDocument(
                                document_id=f"{ticker_name}_{year_name}_{form}_{accno}",
                                document_text=combined_text,
                                query="extract financial information",
                                company=self._get_company_name(ticker_name),
                                ticker=ticker_name,
                                year=year_name,
                                form_type=form,
                                metadata={
                                    "accno": accno,
                                    "source_type": "processed_text",
                                    "segment_count": len(segments),
                                    "original_meta": meta
                                }
                            )
                            
                            documents.append(doc)
                            
                            if limit and len(documents) >= limit:
                                return documents[:limit]
                                
                        except Exception as e:
                            logger.error(f"Error processing {text_file}: {e}")
                            continue
                            
        return documents
    
    def load_with_facts(self, ticker: str = None, year: str = None, 
                       include_facts: bool = True) -> List[FinancialDocument]:
        """Load documents with associated financial facts as ground truth"""
        documents = self.load_processed_text_documents(ticker, year)
        
        if not include_facts:
            return documents
            
        # Enhance with facts data
        for doc in documents:
            facts_file = self._get_facts_file(doc.ticker, doc.year, doc.metadata["accno"])
            if facts_file and facts_file.exists():
                try:
                    facts = self._load_facts_file(facts_file)
                    doc.ground_truth = self._extract_key_facts(facts)
                except Exception as e:
                    logger.error(f"Error loading facts for {doc.document_id}: {e}")
                    
        return documents
    
    def load_test_queries(self) -> List[str]:
        """Load test queries from test_queries.jsonl"""
        queries_file = self.data_root / "test_queries.jsonl"
        queries = []
        
        if queries_file.exists():
            try:
                with open(queries_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            queries.append(data.get("query", ""))
            except Exception as e:
                logger.error(f"Error loading test queries: {e}")
                
        return queries
    
    def convert_to_aie_format(self, documents: List[FinancialDocument], 
                            queries: List[str] = None) -> List[Dict[str, Any]]:
        """Convert to AIE framework JSONL format"""
        aie_documents = []
        
        default_queries = queries or ["extract financial information"]
        
        for doc in documents:
            for query in default_queries:
                aie_doc = {
                    "document_id": doc.document_id,
                    "document_text": doc.document_text,
                    "query": query,
                    "company": doc.company,
                    "ticker": doc.ticker,
                    "year": doc.year,
                    "form_type": doc.form_type,
                    "metadata": doc.metadata
                }
                
                if doc.ground_truth:
                    aie_doc["targets"] = self._convert_ground_truth_to_targets(doc.ground_truth)
                    
                aie_documents.append(aie_doc)
                
        return aie_documents
    
    def save_aie_format(self, documents: List[Dict[str, Any]], output_file: str):
        """Save documents in AIE JSONL format"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def _load_chunks_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line.strip()))
        return chunks
    
    def _load_text_segments(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load text segments from JSONL file"""
        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line.strip()))
        return segments
    
    def _get_facts_file(self, ticker: str, year: str, accno: str) -> Optional[Path]:
        """Get facts file path"""
        facts_path = self.data_root / "processed" / ticker / year / f"{accno.split('_')[0]}_{accno}" / "facts.jsonl"
        return facts_path if facts_path.exists() else None
    
    def _load_facts_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load facts from JSONL file"""
        facts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    facts.append(json.loads(line.strip()))
        return facts
    
    def _extract_key_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key financial facts as ground truth"""
        key_facts = {}
        
        # Mapping of important financial concepts
        fact_mappings = {
            "us-gaap:Revenues": "total_revenue",
            "us-gaap:NetIncomeLoss": "net_income", 
            "us-gaap:Assets": "total_assets",
            "us-gaap:ResearchAndDevelopmentExpense": "rd_expense",
            "us-gaap:CashAndCashEquivalentsAtCarryingValue": "cash_equivalents"
        }
        
        for fact in facts:
            qname = fact.get("qname", "")
            value_num = fact.get("value_num")
            
            if qname in fact_mappings and value_num is not None:
                key_facts[fact_mappings[qname]] = value_num
                
        return key_facts
    
    def _convert_ground_truth_to_targets(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert ground truth to targets format"""
        targets = []
        
        for key, value in ground_truth.items():
            target = {
                "name": key,
                "target_value": value,
                "target_type": "number" if isinstance(value, (int, float)) else "text",
                "unit": "USD" if "revenue" in key or "income" in key or "asset" in key else None
            }
            targets.append(target)
            
        return targets
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker"""
        company_mapping = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "NVDA": "NVIDIA Corporation",
            "META": "Meta Platforms Inc.",
            "TSLA": "Tesla Inc.",
            "BRK-A": "Berkshire Hathaway Inc.",
            "JPM": "JPMorgan Chase & Co.",
        }
        return company_mapping.get(ticker, ticker)


def main():
    """Demo usage of the adapter"""
    adapter = FinancialDataAdapter()
    
    # Load documents from chunked format
    print("Loading chunked documents...")
    chunked_docs = adapter.load_chunked_documents(ticker="AAPL", year="2023", limit=2)
    print(f"Loaded {len(chunked_docs)} chunked documents")
    
    # Load documents with facts
    print("Loading documents with facts...")
    docs_with_facts = adapter.load_with_facts(ticker="AAPL", year="2023")
    print(f"Loaded {len(docs_with_facts)} documents with facts")
    
    # Load test queries
    print("Loading test queries...")
    queries = adapter.load_test_queries()
    print(f"Loaded {len(queries)} test queries")
    
    # Convert to AIE format
    print("Converting to AIE format...")
    aie_docs = adapter.convert_to_aie_format(chunked_docs, queries[:3])
    print(f"Converted to {len(aie_docs)} AIE format documents")
    
    # Save sample
    adapter.save_aie_format(aie_docs, "sample_converted_data.jsonl")
    print("Saved sample converted data")


if __name__ == "__main__":
    main()
