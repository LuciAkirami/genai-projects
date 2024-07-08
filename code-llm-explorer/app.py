import gradio as gr
from huggingface_hub import InferenceClient

css = """
#generate_button {
    transition: background-color 1s ease-out, color 1s ease-out; border-color 1s ease-out;
}
"""

with gr.Blocks(css=css, theme="gradio/soft") as demo:
    gr.Markdown("<center><h1>Code LLM Explorer</h1></center>")

    prompt = gr.Textbox(
        label="Prompt",
        lines=2,  # default two lines length
        max_lines=5,  # the Textbox entends upto 5 lines length
        info="Type your Prompt here",
        show_label=False,
    )
    hf_token = gr.Textbox(
        label="HuggingFace Token",
        type="password",
        placeholder="Your Hugging Face Token",
        show_label=False,
    )

    # gr.Group() will group the two buttons together
    # so there will be no gap between two buttons
    with gr.Group():
        with gr.Row() as button_row:
            # variant: 'primary' for main call-to-action, 'secondary' for a more subdued style, 'stop' for a stop button.
            geneerate_btn = gr.Button(
                "Run", elem_id="generate_button", variant="primary", size="sm"
            )
            view_code = gr.Button(
                "View Code", elem_id="generate_button", variant="secondary", size="sm"
            )

    with gr.Row() as output_row:
        codellama_output = gr.Markdown("meta-llama/CodeLlama-70b-hf")
        stablecode_output = gr.Markdown("stabilityai/stable-code-instruct-3b")
        deepseek_output = gr.Markdown("deepseek-ai/deepseek-coder-33b-instruct")
    pass

demo.launch()
