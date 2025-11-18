"""
Kapittel 7: Tripletex API Klient
Client for integrating with Tripletex accounting system API.
"""
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

try:
    from utils import config, logger, LoggerMixin, SecurityValidator
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin, SecurityValidator


class TripletexError(Exception):
    """Custom exception for Tripletex API errors."""
    pass


class TripletexClient(LoggerMixin):
    """
    Client for Tripletex API integration.
    
    Features:
    - Session token management
    - Customer operations (list, get, create)
    - Invoice operations (list, get, create, send)
    - Product operations
    - Error handling and logging
    """
    
    BASE_URL = "https://api.tripletex.io/v2"
    
    def __init__(
        self,
        employee_token: Optional[str] = None,
        consumer_token: Optional[str] = None
    ):
        """
        Initialize Tripletex client.
        
        Args:
            employee_token: Tripletex employee token
            consumer_token: Tripletex consumer token
        """
        self.employee_token = employee_token or config.tripletex.employee_token
        self.consumer_token = consumer_token or config.tripletex.consumer_token
        self.session_token: Optional[str] = None
        self.session_expires: Optional[datetime] = None
        
        if not self.employee_token or not self.consumer_token:
            raise TripletexError("Missing Tripletex credentials")
        
        self.log_info("Initialized Tripletex client")
    
    def _ensure_session(self):
        """Ensure we have a valid session token."""
        now = datetime.now()
        
        if self.session_token and self.session_expires and now < self.session_expires:
            return  # Session still valid
        
        # Create new session
        url = f"{self.BASE_URL}/token/session/:create"
        params = {
            "consumerToken": self.consumer_token,
            "employeeToken": self.employee_token,
            "expirationDate": (now + timedelta(days=1)).strftime("%Y-%m-%d")
        }
        
        try:
            response = requests.put(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.session_token = data["value"]["token"]
            self.session_expires = datetime.fromisoformat(
                data["value"]["expirationDate"]
            )
            
            self.log_info("Created new Tripletex session")
        except Exception as e:
            raise TripletexError(f"Failed to create session: {e}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Tripletex API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., '/customer')
            params: Query parameters
            json_data: JSON body for POST/PUT
            
        Returns:
            Response data
        """
        self._ensure_session()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.session_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("value", data)
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Tripletex API error: {e}"
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"{error_msg} - {error_data.get('message', '')}"
                except Exception:
                    pass
            self.log_error(error_msg)
            raise TripletexError(error_msg)
        except Exception as e:
            self.log_error(f"Request failed: {e}")
            raise TripletexError(f"Request failed: {e}")
    
    # Customer operations
    
    def list_customers(
        self,
        name: Optional[str] = None,
        is_inactive: bool = False,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List customers.
        
        Args:
            name: Filter by customer name
            is_inactive: Include inactive customers
            count: Maximum results
            
        Returns:
            List of customers
        """
        params = {
            "count": count,
            "isInactive": is_inactive
        }
        if name:
            params["name"] = name
        
        result = self._request("GET", "/customer", params=params)
        return result if isinstance(result, list) else result.get("values", [])
    
    def get_customer(self, customer_id: int) -> Dict[str, Any]:
        """
        Get customer by ID.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Customer data
        """
        return self._request("GET", f"/customer/{customer_id}")
    
    def create_customer(
        self,
        name: str,
        org_number: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create new customer.
        
        Args:
            name: Customer name
            org_number: Organization number
            email: Email address
            phone: Phone number
            **kwargs: Additional customer fields
            
        Returns:
            Created customer data
        """
        # Validate inputs
        SecurityValidator.validate_org_number(org_number)
        if email:
            SecurityValidator.validate_email(email)
        
        data = {
            "name": SecurityValidator.sanitize_input(name),
            "organizationNumber": org_number,
            **kwargs
        }
        if email:
            data["email"] = email
        if phone:
            data["phoneNumber"] = phone
        
        return self._request("POST", "/customer", json_data=data)
    
    # Invoice operations
    
    def list_invoices(
        self,
        customer_id: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List invoices.
        
        Args:
            customer_id: Filter by customer
            from_date: From date (YYYY-MM-DD)
            to_date: To date (YYYY-MM-DD)
            count: Maximum results
            
        Returns:
            List of invoices
        """
        params = {"count": count}
        if customer_id:
            params["customerId"] = customer_id
        if from_date:
            params["invoiceDateFrom"] = from_date
        if to_date:
            params["invoiceDateTo"] = to_date
        
        result = self._request("GET", "/invoice", params=params)
        return result if isinstance(result, list) else result.get("values", [])
    
    def get_invoice(self, invoice_id: int) -> Dict[str, Any]:
        """
        Get invoice by ID.
        
        Args:
            invoice_id: Invoice ID
            
        Returns:
            Invoice data
        """
        return self._request("GET", f"/invoice/{invoice_id}")
    
    def create_invoice(
        self,
        customer_id: int,
        invoice_date: str,
        due_date: str,
        order_lines: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create new invoice.
        
        Args:
            customer_id: Customer ID
            invoice_date: Invoice date (YYYY-MM-DD)
            due_date: Due date (YYYY-MM-DD)
            order_lines: List of order lines
            **kwargs: Additional invoice fields
            
        Returns:
            Created invoice data
        """
        data = {
            "customer": {"id": customer_id},
            "invoiceDate": invoice_date,
            "dueDate": due_date,
            "orderLines": order_lines,
            **kwargs
        }
        
        return self._request("POST", "/invoice", json_data=data)
    
    def send_invoice(
        self,
        invoice_id: int,
        send_type: str = "EMAIL"
    ) -> Dict[str, Any]:
        """
        Send invoice to customer.
        
        Args:
            invoice_id: Invoice ID
            send_type: Send method (EMAIL, EHF, etc.)
            
        Returns:
            Send result
        """
        return self._request(
            "PUT",
            f"/invoice/{invoice_id}/:send",
            params={"sendType": send_type}
        )
    
    # Product operations
    
    def list_products(
        self,
        name: Optional[str] = None,
        is_inactive: bool = False,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List products.
        
        Args:
            name: Filter by product name
            is_inactive: Include inactive products
            count: Maximum results
            
        Returns:
            List of products
        """
        params = {
            "count": count,
            "isInactive": is_inactive
        }
        if name:
            params["name"] = name
        
        result = self._request("GET", "/product", params=params)
        return result if isinstance(result, list) else result.get("values", [])
    
    def get_product(self, product_id: int) -> Dict[str, Any]:
        """
        Get product by ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product data
        """
        return self._request("GET", f"/product/{product_id}")


# Example usage
def example_tripletex_client():
    """Example: Using Tripletex client"""
    try:
        client = TripletexClient()
        
        # List customers
        customers = client.list_customers(count=5)
        print(f"Found {len(customers)} customers")
        for customer in customers[:3]:
            print(f"- {customer.get('name')} (ID: {customer.get('id')})")
        
        # List recent invoices
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        invoices = client.list_invoices(from_date=from_date, count=5)
        print(f"\nRecent invoices: {len(invoices)}")
        for invoice in invoices[:3]:
            print(f"- Invoice #{invoice.get('invoiceNumber')}: {invoice.get('amount')} NOK")
        
        # List products
        products = client.list_products(count=5)
        print(f"\nProducts: {len(products)}")
        for product in products[:3]:
            print(f"- {product.get('name')}: {product.get('priceExcludingVatCurrency')} NOK")
        
    except TripletexError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Tripletex Client Example ===")
    example_tripletex_client()
