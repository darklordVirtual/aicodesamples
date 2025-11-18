"""
Kapittel 13: Fakturabehandling med AI
Automated invoice processing using AI and Tripletex integration.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from utils import config, logger, LoggerMixin, SecurityValidator
    from fundamentals.ai_basics import AIClient
    from mcp.tripletex_client import TripletexClient
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin, SecurityValidator
    from fundamentals.ai_basics import AIClient
    from mcp.tripletex_client import TripletexClient


@dataclass
class InvoiceData:
    """Structured invoice data"""
    invoice_number: str
    date: str
    supplier: str
    customer: str
    line_items: List[Dict[str, Any]]
    subtotal: float
    vat: float
    total: float
    currency: str = "NOK"


class InvoiceProcessor(LoggerMixin):
    """
    AI-powered invoice processing system.
    
    Features:
    - Extract data from invoice text/images
    - Validate extracted data
    - Create invoices in Tripletex
    - Error handling and logging
    """
    
    def __init__(
        self,
        use_tripletex: bool = False
    ):
        self.ai_client = AIClient()
        self.tripletex_client = None
        
        if use_tripletex:
            try:
                self.tripletex_client = TripletexClient()
                self.log_info("Initialized with Tripletex integration")
            except Exception as e:
                self.log_warning(f"Tripletex unavailable: {e}")
        
        self.log_info("Initialized invoice processor")
    
    def extract_invoice_data(self, invoice_text: str) -> InvoiceData:
        """
        Extract structured data from invoice text.
        
        Args:
            invoice_text: Raw invoice text
            
        Returns:
            Structured invoice data
        """
        self.log_info("Extracting invoice data")
        
        schema = {
            "invoice_number": "Invoice number/ID",
            "date": "Invoice date (YYYY-MM-DD)",
            "supplier": "Supplier/vendor name",
            "customer": "Customer name",
            "line_items": "Array of line items with description, quantity, unit_price, total",
            "subtotal": "Subtotal amount (number)",
            "vat": "VAT/tax amount (number)",
            "total": "Total amount (number)",
            "currency": "Currency code (default NOK)"
        }
        
        data = self.ai_client.extract_structured_data(invoice_text, schema)
        
        # Validate and create InvoiceData
        invoice = InvoiceData(
            invoice_number=data.get("invoice_number", "UNKNOWN"),
            date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
            supplier=SecurityValidator.sanitize_input(data.get("supplier", "")),
            customer=SecurityValidator.sanitize_input(data.get("customer", "")),
            line_items=data.get("line_items", []),
            subtotal=float(data.get("subtotal", 0)),
            vat=float(data.get("vat", 0)),
            total=float(data.get("total", 0)),
            currency=data.get("currency", "NOK")
        )
        
        self.log_info(f"Extracted invoice {invoice.invoice_number}")
        return invoice
    
    def validate_invoice(self, invoice: InvoiceData) -> Dict[str, Any]:
        """
        Validate invoice data for correctness.
        
        Args:
            invoice: Invoice data to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not invoice.invoice_number or invoice.invoice_number == "UNKNOWN":
            errors.append("Missing invoice number")
        
        if not invoice.supplier:
            errors.append("Missing supplier name")
        
        if not invoice.customer:
            errors.append("Missing customer name")
        
        # Check calculations
        calculated_total = invoice.subtotal + invoice.vat
        if abs(calculated_total - invoice.total) > 0.01:
            warnings.append(
                f"Total mismatch: {invoice.total} vs calculated {calculated_total}"
            )
        
        # Check line items
        if not invoice.line_items:
            warnings.append("No line items found")
        else:
            line_total = sum(item.get("total", 0) for item in invoice.line_items)
            if abs(line_total - invoice.subtotal) > 0.01:
                warnings.append(
                    f"Line items total {line_total} doesn't match subtotal {invoice.subtotal}"
                )
        
        is_valid = len(errors) == 0
        
        result = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
        
        if is_valid:
            self.log_info(f"Invoice {invoice.invoice_number} validated successfully")
        else:
            self.log_error(f"Invoice validation failed: {errors}")
        
        return result
    
    def create_in_tripletex(
        self,
        invoice: InvoiceData,
        customer_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create invoice in Tripletex.
        
        Args:
            invoice: Invoice data
            customer_id: Tripletex customer ID
            
        Returns:
            Created invoice data or None
        """
        if not self.tripletex_client:
            self.log_error("Tripletex client not available")
            return None
        
        try:
            # Prepare order lines
            order_lines = []
            for item in invoice.line_items:
                order_lines.append({
                    "description": item.get("description", ""),
                    "count": item.get("quantity", 1),
                    "unitPriceExcludingVatCurrency": item.get("unit_price", 0)
                })
            
            # Create invoice
            result = self.tripletex_client.create_invoice(
                customer_id=customer_id,
                invoice_date=invoice.date,
                due_date=invoice.date,  # Should calculate proper due date
                order_lines=order_lines
            )
            
            self.log_info(f"Created invoice in Tripletex: {result.get('id')}")
            return result
            
        except Exception as e:
            self.log_error(f"Failed to create invoice in Tripletex: {e}")
            return None


# Example usage
def example_invoice_processing():
    """Example: Process invoice"""
    processor = InvoiceProcessor(use_tripletex=False)
    
    invoice_text = """
    FAKTURA
    
    Fakturanummer: INV-2025-001
    Dato: 2025-01-20
    
    Fra: Consulting AS
    Org.nr: 123456789
    
    Til: TechCorp Norge
    Org.nr: 987654321
    
    Beskrivelse                  Antall    Pris      Total
    Konsulentarbeid              40 t      1500,-    60 000,-
    Reiseutgifter                1         5 000,-    5 000,-
    
    Subtotal:                                        65 000,-
    MVA (25%):                                       16 250,-
    TOTALT:                                          81 250,-
    
    Betalingsbetingelser: 30 dager
    """
    
    # Extract data
    invoice = processor.extract_invoice_data(invoice_text)
    
    print(f"Invoice: {invoice.invoice_number}")
    print(f"From: {invoice.supplier}")
    print(f"To: {invoice.customer}")
    print(f"Total: {invoice.total} {invoice.currency}")
    print(f"Line items: {len(invoice.line_items)}")
    
    # Validate
    validation = processor.validate_invoice(invoice)
    print(f"\nValidation:")
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")


if __name__ == "__main__":
    print("=== Invoice Processing ===")
    example_invoice_processing()
