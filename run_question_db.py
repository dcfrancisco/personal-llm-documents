import argparse

from initialize_langchain import create_qa_chain, process_llm_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["all", "docs", "code", "logs"],
        default="all",
        help="Limit retrieval to one knowledge source.",
    )
    args = parser.parse_args()
    qa_chain = create_qa_chain(source_type=args.source)

    while True:
        query = input("\nEnter your question (or '/stop' to quit): ")
        if query.lower() == "/stop":
            break

        llm_response = qa_chain.invoke({"query": query})
        process_llm_response(llm_response)


if __name__ == "__main__":
    main()
