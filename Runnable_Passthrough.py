from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()
prompt = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "explain the joke: {joke}",
    input_variables = ["joke"]
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt , model , parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2 , model , parser)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic':'cricket'})
print(result["joke"])
print(result["explaination"])
