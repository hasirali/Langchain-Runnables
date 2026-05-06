from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct", 
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm)
model2= ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}", 
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "Generate a Linkedin Post about {topic}", 
    input_variables = ["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model1, parser),
    'linkedin_post': RunnableSequence(prompt2, model2, parser)
})

print(parallel_chain.invoke({"topic": "AI"}))