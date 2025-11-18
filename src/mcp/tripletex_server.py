"""
Kapittel 7: Tripletex MCP Server
Model Context Protocol server for Tripletex integration.
"""
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from typing import List, Dict, Any
import json

try:
    from utils import logger
    from mcp.tripletex_client import TripletexClient, TripletexError
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import logger
    from mcp.tripletex_client import TripletexClient, TripletexError


def create_tripletex_mcp_server() -> Server:
    """
    Create MCP server for Tripletex integration.
    
    Provides:
    - Resources: Customer and invoice data
    - Tools: Search, create, and manage customers and invoices
    
    Returns:
        Configured MCP Server instance
    """
    server = Server("tripletex-integration")
    
    # Initialize Tripletex client (will be created lazily)
    _client: TripletexClient = None
    
    def get_client() -> TripletexClient:
        """Get or create Tripletex client."""
        nonlocal _client
        if _client is None:
            _client = TripletexClient()
        return _client
    
    # Define resources
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List available Tripletex resources."""
        return [
            Resource(
                uri="tripletex://customers",
                name="Customers",
                mimeType="application/json",
                description="List of all customers in Tripletex"
            ),
            Resource(
                uri="tripletex://invoices",
                name="Invoices",
                mimeType="application/json",
                description="List of recent invoices"
            ),
            Resource(
                uri="tripletex://products",
                name="Products",
                mimeType="application/json",
                description="List of products and services"
            )
        ]
    
    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read Tripletex resource by URI."""
        try:
            client = get_client()
            
            if uri == "tripletex://customers":
                customers = client.list_customers(count=100)
                return json.dumps(customers, indent=2, ensure_ascii=False)
            
            elif uri == "tripletex://invoices":
                invoices = client.list_invoices(count=50)
                return json.dumps(invoices, indent=2, ensure_ascii=False)
            
            elif uri == "tripletex://products":
                products = client.list_products(count=100)
                return json.dumps(products, indent=2, ensure_ascii=False)
            
            elif uri.startswith("tripletex://customer/"):
                customer_id = int(uri.replace("tripletex://customer/", ""))
                customer = client.get_customer(customer_id)
                return json.dumps(customer, indent=2, ensure_ascii=False)
            
            elif uri.startswith("tripletex://invoice/"):
                invoice_id = int(uri.replace("tripletex://invoice/", ""))
                invoice = client.get_invoice(invoice_id)
                return json.dumps(invoice, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
        
        except TripletexError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return json.dumps({"error": f"Internal error: {e}"})
    
    # Define tools
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available Tripletex tools."""
        return [
            Tool(
                name="search_customers",
                description="Search for customers by name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Customer name to search for"
                        },
                        "include_inactive": {
                            "type": "boolean",
                            "description": "Include inactive customers",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="get_customer",
                description="Get detailed customer information by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "integer",
                            "description": "Tripletex customer ID"
                        }
                    },
                    "required": ["customer_id"]
                }
            ),
            Tool(
                name="create_customer",
                description="Create a new customer in Tripletex",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Customer name"
                        },
                        "org_number": {
                            "type": "string",
                            "description": "Norwegian organization number (9 digits)"
                        },
                        "email": {
                            "type": "string",
                            "description": "Customer email address"
                        },
                        "phone": {
                            "type": "string",
                            "description": "Customer phone number"
                        }
                    },
                    "required": ["name", "org_number"]
                }
            ),
            Tool(
                name="list_invoices",
                description="List invoices with optional filters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "integer",
                            "description": "Filter by customer ID"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        }
                    }
                }
            ),
            Tool(
                name="get_invoice",
                description="Get detailed invoice information by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "invoice_id": {
                            "type": "integer",
                            "description": "Tripletex invoice ID"
                        }
                    },
                    "required": ["invoice_id"]
                }
            ),
            Tool(
                name="create_invoice",
                description="Create a new invoice in Tripletex",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "integer",
                            "description": "Customer ID"
                        },
                        "invoice_date": {
                            "type": "string",
                            "description": "Invoice date (YYYY-MM-DD)"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date (YYYY-MM-DD)"
                        },
                        "order_lines": {
                            "type": "array",
                            "description": "Invoice lines",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "integer"},
                                    "description": {"type": "string"},
                                    "count": {"type": "number"},
                                    "unit_price": {"type": "number"}
                                }
                            }
                        }
                    },
                    "required": ["customer_id", "invoice_date", "due_date", "order_lines"]
                }
            ),
            Tool(
                name="send_invoice",
                description="Send invoice to customer via email",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "invoice_id": {
                            "type": "integer",
                            "description": "Invoice ID to send"
                        },
                        "send_type": {
                            "type": "string",
                            "enum": ["EMAIL", "EHF"],
                            "description": "Delivery method",
                            "default": "EMAIL"
                        }
                    },
                    "required": ["invoice_id"]
                }
            ),
            Tool(
                name="list_products",
                description="List products and services",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Filter by product name"
                        },
                        "include_inactive": {
                            "type": "boolean",
                            "description": "Include inactive products",
                            "default": False
                        }
                    }
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute Tripletex tool."""
        try:
            client = get_client()
            
            if name == "search_customers":
                results = client.list_customers(
                    name=arguments.get("name"),
                    is_inactive=arguments.get("include_inactive", False)
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, ensure_ascii=False)
                )]
            
            elif name == "get_customer":
                customer = client.get_customer(arguments["customer_id"])
                return [TextContent(
                    type="text",
                    text=json.dumps(customer, indent=2, ensure_ascii=False)
                )]
            
            elif name == "create_customer":
                customer = client.create_customer(
                    name=arguments["name"],
                    org_number=arguments["org_number"],
                    email=arguments.get("email"),
                    phone=arguments.get("phone")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(customer, indent=2, ensure_ascii=False)
                )]
            
            elif name == "list_invoices":
                invoices = client.list_invoices(
                    customer_id=arguments.get("customer_id"),
                    from_date=arguments.get("from_date"),
                    to_date=arguments.get("to_date")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(invoices, indent=2, ensure_ascii=False)
                )]
            
            elif name == "get_invoice":
                invoice = client.get_invoice(arguments["invoice_id"])
                return [TextContent(
                    type="text",
                    text=json.dumps(invoice, indent=2, ensure_ascii=False)
                )]
            
            elif name == "create_invoice":
                invoice = client.create_invoice(
                    customer_id=arguments["customer_id"],
                    invoice_date=arguments["invoice_date"],
                    due_date=arguments["due_date"],
                    order_lines=arguments["order_lines"]
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(invoice, indent=2, ensure_ascii=False)
                )]
            
            elif name == "send_invoice":
                result = client.send_invoice(
                    invoice_id=arguments["invoice_id"],
                    send_type=arguments.get("send_type", "EMAIL")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
            
            elif name == "list_products":
                products = client.list_products(
                    name=arguments.get("name"),
                    is_inactive=arguments.get("include_inactive", False)
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(products, indent=2, ensure_ascii=False)
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except TripletexError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Internal error: {e}"})
            )]
    
    return server


# Example usage
def example_tripletex_mcp():
    """Example: Using Tripletex MCP server"""
    import asyncio
    
    async def run_example():
        server = create_tripletex_mcp_server()
        
        # List resources
        resources = await server.list_resources()
        print(f"Available resources: {len(resources)}")
        for res in resources:
            print(f"- {res.name}: {res.description}")
        
        # List tools
        tools = await server.list_tools()
        print(f"\nAvailable tools: {len(tools)}")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        # Search customers
        try:
            result = await server.call_tool(
                name="search_customers",
                arguments={"name": "AS"}
            )
            print(f"\nCustomer search result:\n{result[0].text[:200]}...")
        except Exception as e:
            print(f"\nNote: Requires valid Tripletex credentials: {e}")
    
    asyncio.run(run_example())


if __name__ == "__main__":
    print("=== Tripletex MCP Server ===")
    example_tripletex_mcp()
