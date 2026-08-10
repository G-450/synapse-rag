import os
import json
from datasets import load_dataset

def main():
    print("Downloading LegalBench-RAG dataset...")
    try:
        ds = load_dataset("amentaphd/legalbench-qa", "all")
    except Exception as e:
        print(f"Failed to load 'all', trying default: {e}")
        ds = load_dataset("amentaphd/legalbench-qa")
    
    print(ds)
    
    split = 'train' if 'train' in ds else list(ds.keys())[0]
    
    documents = []
    for i, row in enumerate(ds[split]):
        if i >= 50: # Limit to 50 for quick prototyping
            break
        
        doc_text = row.get('text', row.get('document', row.get('content', str(row))))
        doc_id = row.get('id', row.get('doc_id', f"doc_{i}"))
        
        documents.append({
            "id": str(doc_id),
            "text": doc_text,
            "metadata": {k: v for k, v in row.items() if k not in ['text', 'document', 'content']}
        })
        
    os.makedirs('data', exist_ok=True)
    with open('data/legalbench.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
        
    print(f"Successfully saved {len(documents)} documents to data/legalbench.json")

if __name__ == "__main__":
    main()
