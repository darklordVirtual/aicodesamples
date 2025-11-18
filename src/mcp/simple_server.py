"""
Kapittel 6: Enkel MCP Server
Simple Model Context Protocol server for customer database access.
"""
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from typing import List, Dict, Any, Optional
import json

try:
    from utils import logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import logger, LoggerMixin


class CustomerDatabase(LoggerMixin):
    """
    Simple in-memory customer database for MCP demo.
    """
    
    def __init__(self):
        self.customers = {
            "1": {
                "id": "1",
                "name": "Acme AS",
                "org_number": "123456789",
                "contact_email": "post@acme.no",
                "status": "active"
            },
            "2": {
                "id": "2",
                "name": "TechCorp Norge",
                "org_number": "987654321",
                "contact_email": "kontakt@techcorp.no",
                "status": "active"
            },
            "3": {
                "id": "3",
                "name": "Nordic Solutions",
                "org_number": "555666777",
                "contact_email": "info@nordic.no",
                "status": "inactive"
            }
        }
        self.log_info(f"Initialized customer database with {len(self.customers)} customers")
    
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID."""
        return self.customers.get(customer_id)
    
    def search_customers(
        self,
        query: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search customers by name or filter by status."""
        results = list(self.customers.values())
        
        if status:
            results = [c for c in results if c.get("status") == status]
        
        if query:
            query_lower = query.lower()
            results = [
                c for c in results
                if query_lower in c.get("name", "").lower() or
                   query_lower in c.get("org_number", "")
            ]
        
        return results
    
    def add_customer(
        self,
        name: str,
        org_number: str,
        contact_email: str,
        status: str = "active"
    ) -> Dict[str, Any]:
        """Add new customer."""
        customer_id = str(len(self.customers) + 1)
        customer = {
            "id": customer_id,
            "name": name,
            "org_number": org_number,
            "contact_email": contact_email,
            "status": status
        }
        self.customers[customer_id] = customer
        self.log_info(f"Added customer: {name}")
        return customer
    
    def update_customer(
        self,
        customer_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update customer fields."""
        if customer_id not in self.customers:
            return None
        
        self.customers[customer_id].update(updates)
        self.log_info(f"Updated customer {customer_id}")
        return self.customers[customer_id]


def create_customer_mcp_server() -> Server:
    """
    Create MCP server for customer database.
    
    Returns:
        Configured MCP Server instance
    """
    server = Server("customer-database")
    db = CustomerDatabase()
    
    # Define resources (read-only data)
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List available customer resources."""
        resources = []
        for customer_id, customer in db.customers.items():
            resources.append(
                Resource(
                    uri=f"customer://{customer_id}",
                    name=f"Customer: {customer['name']}",
                    mimeType="application/json",
                    description=f"Customer data for {customer['name']} (ID: {customer_id})"
                )
            )
        return resources
    
    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read customer resource by URI."""
        # Extract customer ID from URI (customer://123)
        if not uri.startswith("customer://"):
            raise ValueError(f"Invalid URI: {uri}")
        
        customer_id = uri.replace("customer://", "")
        customer = db.get_customer(customer_id)
        
        if not customer:
            raise ValueError(f"Customer not found: {customer_id}")
        
        return json.dumps(customer, indent=2)
    
    # Define tools (operations that modify data)
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available customer tools."""
        return [
            Tool(
                name="search_customers",
                description="Search for customers by name or org number, optionally filter by status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (name or org number)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive"],
                            "description": "Filter by customer status"
                        }
                    }
                }
            ),
            Tool(
                name="get_customer",
                description="Get customer details by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID"
                        }
                    },
                    "required": ["customer_id"]
                }
            ),
            Tool(
                name="add_customer",
                description="Add a new customer",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Customer name"
                        },
                        "org_number": {
                            "type": "string",
                            "description": "Organization number"
                        },
                        "contact_email": {
                            "type": "string",
                            "description": "Contact email"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive"],
                            "description": "Customer status"
                        }
                    },
                    "required": ["name", "org_number", "contact_email"]
                }
            ),
            Tool(
                name="update_customer",
                description="Update customer information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID to update"
                        },
                        "name": {"type": "string"},
                        "org_number": {"type": "string"},
                        "contact_email": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive"]
                        }
                    },
                    "required": ["customer_id"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute customer database tool."""
        if name == "search_customers":
            results = db.search_customers(
                query=arguments.get("query"),
                status=arguments.get("status")
            )
            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2)
            )]
        
        elif name == "get_customer":
            customer = db.get_customer(arguments["customer_id"])
            if not customer:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Customer not found"})
                )]
            return [TextContent(
                type="text",
                text=json.dumps(customer, indent=2)
            )]
        
        elif name == "add_customer":
            customer = db.add_customer(
                name=arguments["name"],
                org_number=arguments["org_number"],
                contact_email=arguments["contact_email"],
                status=arguments.get("status", "active")
            )
            return [TextContent(
                type="text",
                text=json.dumps(customer, indent=2)
            )]
        
        elif name == "update_customer":
            customer_id = arguments.pop("customer_id")
            customer = db.update_customer(customer_id, **arguments)
            if not customer:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Customer not found"})
                )]
            return [TextContent(
                type="text",
                text=json.dumps(customer, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    return server


# Example usage
def example_mcp_server():
    """Example: Create and use MCP server"""
    import asyncio
    
    async def run_example():
        server = create_customer_mcp_server()
        
        # List resources
        resources = await server.list_resources()
        print(f"Available resources: {len(resources)}")
        for res in resources:
            print(f"- {res.name} ({res.uri})")
        
        # List tools
        tools = await server.list_tools()
        print(f"\nAvailable tools: {len(tools)}")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        # Call search tool
        result = await server.call_tool(
            name="search_customers",
            arguments={"status": "active"}
        )
        print(f"\nActive customers:\n{result[0].text}")
        
        # Add customer
        result = await server.call_tool(
            name="add_customer",
            arguments={
                "name": "New Company AS",
                "org_number": "111222333",
                "contact_email": "new@company.no"
            }
        )
        print(f"\nAdded customer:\n{result[0].text}")
    
    asyncio.run(run_example())


if __name__ == "__main__":
    print("=== MCP Server Example ===")
    example_mcp_server()
