"""
Del VI: Case Studies
Practical examples: invoice processing, customer support, multimodal, and ethics.
"""

from .invoice_processing import InvoiceProcessor, InvoiceData
from .customer_support import CustomerSupportBot, SupportTicket
from .multimodal import ImageAnalyzer, DocumentAnalyzer
from .ethics import EthicsChecker, BiasDetector

__all__ = [
    "InvoiceProcessor",
    "InvoiceData",
    "CustomerSupportBot",
    "SupportTicket",
    "ImageAnalyzer",
    "DocumentAnalyzer",
    "EthicsChecker",
    "BiasDetector",
]
