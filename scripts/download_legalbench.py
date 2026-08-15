import argparse
import json
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download LegalBench-RAG dataset for Synapse RAG evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to download (default: None to download all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/legalbench.json",
        help="Destination path for the output JSON file (default: data/legalbench.json)",
    )
    args = parser.parse_args()

    print("Downloading LegalBench-RAG dataset...")
    try:
        ds = load_dataset("amentaphd/legalbench-qa", "all")
    except Exception as e:
        print(f"Failed to load 'all', trying default: {e}")
        ds = load_dataset("amentaphd/legalbench-qa")

    print(ds)

    split = "train" if "train" in ds else list(ds.keys())[0]

    documents = []

    for i, row in enumerate(ds[split]):
        if args.limit is not None and i >= args.limit:
            print(f"Reached limit of {args.limit} documents.")
            break

        doc_text = row.get("text", row.get("document", row.get("content", str(row))))
        doc_id = row.get("id", row.get("doc_id", f"doc_{i}"))

        documents.append({
            "id": str(doc_id),
            "text": doc_text,
            "metadata": {
                k: v
                for k, v in row.items()
                if k not in ["text", "document", "content"]
            },
        })

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print(f"Successfully saved {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()
