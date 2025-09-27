"""
AIE Framework Core Modules

Contains four core components:
- Segmentation: Document segmentation
- Retrieval: Relevant segment retrieval
- Summarization: Summary generation
- Extraction: Information extraction
"""

from .segmentation import DocumentSegmenter, DocumentSegment
from .retrieval import DocumentRetriever, RetrievalResult
from .summarization import DocumentSummarizer, SummaryResult
from .extraction import InformationExtractor, ExtractionTarget, ExtractionResult
from .pipeline import AIEPipeline, AIEPipelineResult

__all__ = [
    "DocumentSegmenter",
    "DocumentSegment",
    "DocumentRetriever", 
    "RetrievalResult",
    "DocumentSummarizer",
    "SummaryResult",
    "InformationExtractor",
    "ExtractionTarget",
    "ExtractionResult",
    "AIEPipeline",
    "AIEPipelineResult"
]
