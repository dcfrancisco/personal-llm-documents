import gradio as gr
from initialize_langchain import create_qa_chain, html_llm_response

qa_chain = create_qa_chain()


# Function to process user input and generate response
def generate_response(query):
    # Prompt for query and display response
    llm_response = qa_chain.invoke({"query": query})
    response = html_llm_response(llm_response)
    print(response)
    return response


# Set up the Gradio user interface
inputs = gr.inputs.Textbox(lines=2, placeholder="Enter your question here...")
outputs = gr.outputs.HTML()

interface = gr.Interface(fn=generate_response, inputs=inputs, outputs=outputs)

# Run the Gradio interface
interface.launch()
