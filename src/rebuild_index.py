from rag import build_index


def main():
    result = build_index()
    print(f"Saved {result['chunkCount']} chunks to ai/index/rag-index.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        raise SystemExit(1) from error
