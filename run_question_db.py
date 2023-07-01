from initialize_langchain import qa_chain, process_llm_response

# Prompt for query and display response
while True:
    query = input("\nEnter your question (or '/stop' to quit): ")
    if query.lower() == "/stop":
        break

    llm_response = qa_chain(query)
    process_llm_response(llm_response)
