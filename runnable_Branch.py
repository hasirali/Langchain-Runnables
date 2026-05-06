from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

# Prompt to generate report
prompt1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt to summarize
prompt2 = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Output parser
parser = StrOutputParser()

# Step 1: Generate report
report_gen_chain = RunnableSequence(prompt1, model, parser)

# Step 2: Branch (if long → summarize, else → return as is)
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 100,
        RunnableLambda(lambda x: {"text": x}) | RunnableSequence(prompt2, model, parser)
    ),
    RunnablePassthrough()
)

# Final chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Run
result = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)