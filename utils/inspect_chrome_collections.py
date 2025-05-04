#!/usr/bin/env python
"""
Utility script for managing ChromaDB collections.

This script provides command-line functionalities to list collections, delete a specific collection, or retrieve metadata for a collection in ChromaDB.

Dependencies:
- chromadb: For interacting with ChromaDB collections.
- argparse: For command-line argument parsing.

Usage:
    python script_name.py <list|delete|model> [--name COLLECTION_NAME] [--path PERSIST_DIR]
"""
import argparse
import textwrap

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


# Initialize the ChromaDB client
def initialize_chroma_client(persist_directory):
    return chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# List all collections
def list_collections(client):
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
    else:
        for collection in collections:
            print(collection)
            #print(f"Model: {collection.get_model()}")
            #print(collection.metadata)
            #print(collection.database)
            #print(f"Configure: {collection.configuration_json}")
            print("--- %% ---")
            #print(f"Collection Name: {collection['name']}")
            #print(f"Metadata: {collection.get('metadata', {})}\n")

# Delete a collection
def delete_collection(client, name):
    client.delete_collection(name=name)
    print(f"Deleted Collection: {name}")

# Get metadata for a specific collection
def get_collection_model(client, name):
    try:
        collection = client.get_collection(name=name)
        print(f"Collection Model:\n{collection.get_model()}")
        #print(f"Metadata: {collection.metadata}")
    except Exception as e:
        print(f"Error fetching metadata for collection '{name}': {e}")

# Get collection stats
def get_collection_stats(client, name):
    try:
        collection = client.get_collection(name=name)
        data = collection.get(
            include=["metadatas", "documents", "embeddings"],
        )
        print(f"Collection Stats:\n\tEmbedding count: {len(data['embeddings'])}\n\tDocument count: {len(data['documents'])}\n\tMetadata count: {len(data['metadatas'])}")

    except Exception as e:
        print(f"Error fetching stats for collection '{name}': {e}")

def get_collection_sample(client, name, index=0, char_limit=100):
    try:
        collection = client.get_collection(name=name)
        data = collection.get(
            include=["metadatas", "documents", "embeddings"],
        )
        if char_limit == 0:
            print("Print raw data")
            print(f"Show Sample data[{index}] in raw data format\n"
                  f"--- Documents ---\n{data['documents'][index]}\n"
                  f"\n--- Embeddings ---\n{data['embeddings'][index]}\n"
                  f"\n--- Metadata ---\n{data['metadatas'][index]}\n")
        else:
            doc_preview = data['documents'][index][:char_limit]
            embedding_preview = str(data['embeddings'][index])[:char_limit]

            print(f"Show Sample data[{index}], limited by {char_limit}\n"
                  f"--- Documents ---\n{textwrap.fill(doc_preview, width=80)}\n...\n"
                  f"\n--- Embedding ---\n{embedding_preview}\n...\n"
                  f"\n--- Metadata ---\n{data['metadatas'][index]}\n")
    except Exception as e:
        print(f"Error fetching stats for collection '{name}': {e}")

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Manage ChromaDB collections.")
    parser.add_argument(
        "action",
        choices=["list", "delete", "model", "stats", "sample"],
        help="Action to perform: list, delete, model, stats, sample",
    )
    parser.add_argument(
        "--path",
        default="./db_store",
        help="Path to the persistent directory for ChromaDB (default: ./db_store).",
    )
    parser.add_argument(
        "--name",
        help="Name of the collection (required for delete, model, stats, sample).")
    parser.add_argument(
        "--char-limit",
        type=int,
        default=100,
        help="Maximum number of characters to display for each field (default: 100). To view raw data, set to 0.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the sample to retrieve from the collection (default: 0, required for 'sample').",
    )

    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Initialize Chroma client
    chroma_client = initialize_chroma_client(persist_directory=args.path)

    if args.action == "list":
        list_collections(chroma_client)

    elif args.action == "delete":
        if not args.name:
            print("Error: --name is required for deleting a collection.")
            return
        delete_collection(chroma_client, name=args.name)

    elif args.action == "model":
        if not args.name:
            print("Error: --name is required to fetch collection metadata.")
            return
        get_collection_model(chroma_client, name=args.name)

    elif args.action == "stats":
        if not args.name:
            print("Error: --name is required to fetch collection stats.")
            return
        get_collection_stats(chroma_client, name=args.name)

    elif args.action == "sample":
        if not args.name:
            print("Error: --name is required to fetch collection sample.")
            return
        get_collection_sample(chroma_client, name=args.name, index=args.index, char_limit=args.char_limit)

if __name__ == "__main__":
    main()
