import gradio as gr
from initialize_langchain import qa_chain, html_llm_response


# Function to generate response using LangChain
def generate_response(message):
    # Prompt for query and display response
    query = message
    llm_response = qa_chain(query)
    response = html_llm_response(llm_response)
    return response


# Set up the Gradio user interface
msg = gr.inputs.Textbox(lines=2, placeholder="Enter your message here...")
output_html = gr.outputs.HTML()

# Create a Gradio interface
interface = gr.Interface(fn=generate_response, inputs=msg, outputs=output_html)
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history=[]):
        chat_history.append(message)
        bot_message = generate_response(message)
        chat_history.append(bot_message)
        return bot_message, chat_history

    submit = gr.SubmitButton(label="Send")

# Launch the Gradio interface
interface.launch()
