"""
Kapittel 5: ChromaDB Backup og Gjenoppretting
Utilities for backing up and restoring ChromaDB collections.
"""
import chromadb
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from utils import logger, LoggerMixin
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import logger, LoggerMixin


class BackupManager(LoggerMixin):
    """
    Manage ChromaDB collection backups.
    """
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self.log_info(f"Backup manager initialized with directory: {backup_dir}")
    
    def backup_collection(
        self,
        collection: chromadb.Collection,
        backup_name: Optional[str] = None
    ) -> str:
        """
        Backup a ChromaDB collection to JSON file.
        
        Args:
            collection: ChromaDB collection to backup
            backup_name: Optional backup name (default: collection_name_timestamp)
            
        Returns:
            Path to backup file
        """
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{collection.name}_{timestamp}"
        
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.json")
        
        # Get all documents
        all_data = collection.get()
        
        # Prepare backup data
        backup_data = {
            "collection_name": collection.name,
            "collection_metadata": collection.metadata,
            "backup_timestamp": datetime.now().isoformat(),
            "document_count": len(all_data["ids"]),
            "documents": []
        }
        
        # Add each document
        for i, doc_id in enumerate(all_data["ids"]):
            doc = {
                "id": doc_id,
                "document": all_data["documents"][i],
                "metadata": all_data["metadatas"][i] if all_data["metadatas"] else None,
                "embedding": all_data["embeddings"][i] if all_data.get("embeddings") else None
            }
            backup_data["documents"].append(doc)
        
        # Write to file
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        self.log_info(f"Backed up {len(all_data['ids'])} documents to {backup_path}")
        return backup_path
    
    def restore_collection(
        self,
        client: chromadb.Client,
        backup_path: str,
        collection_name: Optional[str] = None
    ) -> chromadb.Collection:
        """
        Restore a ChromaDB collection from backup.
        
        Args:
            client: ChromaDB client
            backup_path: Path to backup JSON file
            collection_name: Optional new collection name (default: original name)
            
        Returns:
            Restored collection
        """
        # Load backup
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        # Use original name if not specified
        if not collection_name:
            collection_name = backup_data["collection_name"]
        
        # Create or get collection
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata=backup_data.get("collection_metadata", {})
            )
        except Exception:
            # Collection exists, get it
            collection = client.get_collection(name=collection_name)
        
        # Restore documents in batches
        batch_size = 100
        documents = backup_data["documents"]
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            ids = [doc["id"] for doc in batch]
            docs = [doc["document"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch if doc["metadata"]]
            embeddings = [doc["embedding"] for doc in batch if doc.get("embedding")]
            
            # Add batch
            add_kwargs = {
                "ids": ids,
                "documents": docs
            }
            if metadatas:
                add_kwargs["metadatas"] = metadatas
            if embeddings:
                add_kwargs["embeddings"] = embeddings
            
            collection.add(**add_kwargs)
        
        self.log_info(f"Restored {len(documents)} documents to collection '{collection_name}'")
        return collection
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.backup_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    backups.append({
                        "filename": filename,
                        "collection_name": data.get("collection_name"),
                        "timestamp": data.get("backup_timestamp"),
                        "document_count": data.get("document_count"),
                        "size_bytes": os.path.getsize(filepath)
                    })
                except Exception as e:
                    self.log_error(f"Error reading backup {filename}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return backups
    
    def delete_backup(self, backup_filename: str) -> bool:
        """
        Delete a backup file.
        
        Args:
            backup_filename: Name of backup file to delete
            
        Returns:
            True if successful
        """
        filepath = os.path.join(self.backup_dir, backup_filename)
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self.log_info(f"Deleted backup: {backup_filename}")
                return True
            else:
                self.log_warning(f"Backup not found: {backup_filename}")
                return False
        except Exception as e:
            self.log_error(f"Error deleting backup: {e}")
            return False


# Convenience functions
def backup_collection(
    collection: chromadb.Collection,
    backup_path: str
) -> bool:
    """
    Quick backup function.
    
    Args:
        collection: Collection to backup
        backup_path: Path for backup file
        
    Returns:
        True if successful
    """
    try:
        # Get all data
        all_data = collection.get()
        
        backup_data = {
            "collection_name": collection.name,
            "collection_metadata": collection.metadata,
            "backup_timestamp": datetime.now().isoformat(),
            "document_count": len(all_data["ids"]),
            "documents": []
        }
        
        for i, doc_id in enumerate(all_data["ids"]):
            doc = {
                "id": doc_id,
                "document": all_data["documents"][i],
                "metadata": all_data["metadatas"][i] if all_data["metadatas"] else None
            }
            backup_data["documents"].append(doc)
        
        # Write backup
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def restore_collection(
    client: chromadb.Client,
    backup_path: str,
    collection_name: Optional[str] = None
) -> Optional[chromadb.Collection]:
    """
    Quick restore function.
    
    Args:
        client: ChromaDB client
        backup_path: Path to backup file
        collection_name: Optional collection name
        
    Returns:
        Restored collection or None on error
    """
    try:
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        if not collection_name:
            collection_name = backup_data["collection_name"]
        
        # Create collection
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata=backup_data.get("collection_metadata", {})
            )
        except Exception:
            collection = client.get_collection(name=collection_name)
        
        # Restore documents
        documents = backup_data["documents"]
        if documents:
            ids = [doc["id"] for doc in documents]
            docs = [doc["document"] for doc in documents]
            metadatas = [doc["metadata"] or {} for doc in documents]
            
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metadatas
            )
        
        return collection
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return None


# Example usage
def example_backup_restore():
    """Example: Backup and restore operations"""
    from vector_db.chromadb_basics import KnowledgeBase
    
    # Create knowledge base with data
    kb = KnowledgeBase(collection_name="backup_test")
    
    docs = [
        "Python er et programmeringsspråk",
        "JavaScript brukes for webutvikling",
        "ChromaDB er en vektordatabase"
    ]
    metadatas = [
        {"category": "programming"},
        {"category": "web"},
        {"category": "database"}
    ]
    kb.add_batch(docs, metadatas)
    
    print(f"Original collection: {kb.count()} documents")
    
    # Backup
    manager = BackupManager()
    backup_path = manager.backup_collection(kb.collection, "test_backup")
    print(f"\nBackup created: {backup_path}")
    
    # List backups
    backups = manager.list_backups()
    print(f"\nAvailable backups: {len(backups)}")
    for backup in backups:
        print(f"- {backup['filename']}: {backup['document_count']} docs")
    
    # Clear original
    kb.clear()
    print(f"\nAfter clear: {kb.count()} documents")
    
    # Restore
    client = chromadb.Client()
    restored = manager.restore_collection(client, backup_path, "restored_test")
    print(f"\nRestored collection: {restored.count()} documents")
    
    # Verify
    result = restored.get()
    print("\nRestored documents:")
    for doc in result["documents"]:
        print(f"- {doc}")


if __name__ == "__main__":
    print("=== Backup and Restore ===")
    example_backup_restore()
