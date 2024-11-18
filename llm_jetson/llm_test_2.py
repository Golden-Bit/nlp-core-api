from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

####

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)

print(ai_msg.content)


# Classe per mantenere il contesto della conversazione
class TerminalChatbot:
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.messages = [SystemMessage(content="You're a helpful assistant")]

    def chat(self, user_input):
        # Aggiungi il messaggio umano alla conversazione
        self.messages.append(HumanMessage(content=user_input))

        # Ottieni la risposta del modello
        ai_message = self.chat_model.invoke(self.messages)

        # Aggiungi la risposta AI al contesto
        self.messages.append(ai_message)

        return ai_message.content


# Avvia il chatbot
if __name__ == "__main__":
    chatbot = TerminalChatbot(chat_model)
    print("Chatbot avviato. Digita 'exit' per terminare la chat.")
    while True:
        # Input dell'utente
        user_input = input("Tu: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat terminata.")
            break
        # Genera la risposta
        response = chatbot.chat(user_input)
        print(f"AI: {response}")

