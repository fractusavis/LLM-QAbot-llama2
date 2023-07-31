from llama_cpp import Llama

llm = Llama(model_path="./model/llama-2-13b-chat.ggmlv3.q4_K_M.bin")

while True:
    input_text = input('> ')

    output = llm(
        "Q: " + input_text + " A: ",
        max_tokens=32, # increse if you need long answer
        stop=["Q:", "\n"], # comment out if you need long answer
        echo=True,
    )

    print(output['choices'][0]['text'])