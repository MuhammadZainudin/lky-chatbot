import gradio as gr
from src.chat import ask, source_list

DISCLAIMER = ("⚠️ AI simulation built with RAG on Lee Kuan Yew's public speeches/writings. "
              "Responses are NOT real quotes and may extrapolate.")


def respond(message, history):
    answer, hits = ask(message, history)
    srcs = source_list(hits)
    if srcs:
        answer += "\n\n---\n**Sources retrieved:**\n" + "\n".join(f"- {s}" for s in srcs)
    return answer


demo = gr.ChatInterface(
    respond,
    type="messages",
    title="What Would Lee Kuan Yew Do?",
    description=DISCLAIMER,
    examples=[
        "What is your view on democracy versus economic development?",
        "How should a small country survive between great powers?",
        "What advice would you give a young person choosing a career?",
        "What do you think about AI and ChatGPT?",
    ],
)

if __name__ == "__main__":
    demo.launch()
