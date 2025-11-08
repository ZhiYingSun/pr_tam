"""
Mock IncorporationSearcher for development and testing
"""
from typing import List
from src.data.models import BusinessRecord


class MockIncorporationSearcher:
    """Mock version of IncorporationSearcher for testing without API calls"""
    
    def __init__(self, zyte_api_key: str = "mock_key"):
        self.zyte_api_key = zyte_api_key
    
    def search_business(self, business_name: str, limit: int = 5) -> List[BusinessRecord]:
        """Return mock business records for testing"""
        
        # Create mock data based on the business name
        mock_records = []
        
        # Generate 2-3 mock records
        for i in range(min(3, limit)):
            mock_record = BusinessRecord(
                legal_name=f"{business_name} Corp {i+1}",
                registration_number=f"12345{i+1}",
                registration_index=f"12345{i+1}-111",
                status="ACTIVE",
                business_address=f"{100+i} Main St, San Juan, 00908",
                resident_agent_name=f"Agent {i+1}",
                resident_agent_address=f"{200+i} Agent St"
            )
            mock_records.append(mock_record)
        
        return mock_records
    
    def get_business_details(self, registry_number: str) -> BusinessRecord:
        """Return mock business details for testing"""
        return BusinessRecord(
            legal_name=f"Mock Business {registry_number}",
            registration_number=registry_number,
            registration_index=f"{registry_number}-111",
            status="ACTIVE",
            business_address="123 Mock Street, San Juan, 00908",
            resident_agent_name="Mock Agent",
            resident_agent_address="456 Agent Avenue"
        )
