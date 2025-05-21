# !pip install langchain langchain_ollama
# !pip install langchain_openai
#
#
# ###### First Block ######
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
#
# # Initialize the LLM with OpenRouter
# llm = ChatOpenAI(
#     model="meta-llama/llama-3.2-1b-instruct:free",  # Choose your preferred Llama model
#     openai_api_key="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",
#     openai_api_base="https://openrouter.ai/api/v1",
#     temperature=0.7,
# )
#
# # Create a prompt template with system instructions
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that engages in interactive conversations. Ask follow-up questions when you need more information to provide a complete answer."),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{input}")
# ])
#
# # Set up memory to store conversation history
# memory = ConversationBufferMemory(return_messages=True, memory_key="history")
#
# # Create the conversation chain
# conversation = ConversationChain(
#     llm=llm,
#     prompt=prompt,
#     memory=memory,
#     verbose=True
# )
#
# # Function to interact with the model
# def chat_with_llm(user_input):
#     response = conversation.predict(input=user_input)
#     return response
#
#
# ###### Second Block ######
# def chat_session():
#     print("Welcome to your chatbot! Type 'exit' to end the conversation.")
#
#     # Main conversation loop
#     while True:
#         # Prompt the user for input
#         user_input = input("You: ")
#
#         # Check for exit condition
#         if user_input.lower() in ["exit", "quit", ":q"]:
#             print("Goodbye! Thanks for chatting.")
#             break
#
#         # Process the input and get response from your model
#         response = chat_with_llm(user_input)
#
#         # Display the response
#         print(f"Bot: {response}")
#
#
# # Start the chat session
# if __name__ == "__main__":
#     chat_session()
