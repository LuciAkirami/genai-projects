import os
import gradio as gr
from functools import partial
from huggingface_hub import InferenceClient

css = """
#generate_button {
    transition: background-color 1s ease-out, color 1s ease-out; border-color 1s ease-out;
}
"""


def generate(prompt: str, hf_token: str, model: str):
    messages = [{"role": "user", "content": prompt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
    client = InferenceClient(model, token=hf_token)
    model_name = model.split("/")[1]
    response = f"**{model_name}**\n\n"
    for msg in client.chat_completion(messages, max_tokens=600, stream=True):
        token = msg.choices[0].delta.content
        response += token
        yield response


def clear_token():
    # returns a textbox with visibility set to False
    # this will update the hf_token widget thus hiding it
    return gr.Textbox(visible=False)


with gr.Blocks(css=css, theme="gradio/soft") as demo:
    gr.Markdown("<center><h1>Code LLM Explorer</h1></center>")

    prompt = gr.Textbox(
        label="Prompt",
        lines=2,  # default two lines length
        max_lines=5,  # the Textbox entends upto 5 lines length
        info="Type your Prompt here",
        show_label=False,
        value="Write Bubble Sort in Python",
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
            generate_btn = gr.Button(
                "Run", elem_id="generate_button", variant="primary", size="sm"
            )
            view_code = gr.Button(
                "View Code", elem_id="generate_button", variant="secondary", size="sm"
            )

    with gr.Row() as output_row:
        codellama_output = gr.Markdown("codellama/CodeLlama-34b-Instruct-hf")
        stablecode_output = gr.Markdown("stabilityai/stable-code-instruct-3b")
        phi_output = gr.Markdown("microsoft/Phi-3-mini-4k-instruct")

    # Upgrade 1
    # this works great, but what if the user wants to press shit+enter after
    # writing the prompt instead of Run button. Then this fails
    # generate_btn.click(
    #     generate, prompt, [codellama_output, stablecode_output, deepseek_output]
    # )

    # Upgrade 2
    # this works when the user presses shit+enter after writing the prompt
    # the shit+enter will trigger prompt submit
    # gr.on() sets up an event listener which listens for the events given
    # in the list
    # gr.on(
    #     # here either if user presses shit+enter or uses clicks on the
    #     # run button, the inference starts, so we mention both in the list
    #     [prompt.submit, generate_btn.click],
    #     # here we call partial function to partially initialize the geberate
    #     # with the variable model, this is because the below doesnt work
    #     # fn = generate(model="Hey") # gives an error
    #     fn=partial(generate, model="Hey"),
    #     inputs=[prompt, hf_token],
    #     outputs=[codellama_output, stablecode_output, deepseek_output],
    # )

    # Upgrade 3
    # Here, first as soon as user enters a prompt and hf_token and presses the
    # shift+enter or run button, the hf_token widget gets hidden
    # after that we call .then() method to call the inference function
    # gr.on(
    #     [prompt.submit, generate_btn.click], clear_token, inputs=None, outputs=hf_token
    # ).then(
    #     fn=partial(generate, model="Hey"),
    #     inputs=[prompt, hf_token],
    #     outputs=[codellama_output, stablecode_output, deepseek_output],
    # )

    # Upgrade 4
    # Writing the code in a way that each output markdown is handled separetely
    gr.on(
        [prompt.submit, generate_btn.click], clear_token, inputs=None, outputs=hf_token
    ).then(
        fn=partial(generate, model="codellama/CodeLlama-34b-Instruct-hf"),
        inputs=[prompt, hf_token],
        outputs=codellama_output,
    )

    gr.on(
        [prompt.submit, generate_btn.click], clear_token, inputs=None, outputs=hf_token
    ).then(
        fn=partial(generate, model="stabilityai/stable-code-instruct-3b"),
        inputs=[prompt, hf_token],
        outputs=stablecode_output,
    )

    gr.on(
        [prompt.submit, generate_btn.click], clear_token, inputs=None, outputs=hf_token
    ).then(
        fn=partial(generate, model="microsoft/Phi-3-mini-4k-instruct"),
        inputs=[prompt, hf_token],
        outputs=phi_output,
    )

demo.launch()
