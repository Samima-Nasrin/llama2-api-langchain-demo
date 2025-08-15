from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"

app = FastAPI(title="LangChain Server")

class EssayRequest(BaseModel):
    topic: str

prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} within 100 words.")
llm = OllamaLLM(model="gemma3:1b")

@app.post("/essay")
async def generate_essay(request: EssayRequest):
    try:
        # Use keyword argument instead of dict
        formatted_prompt = prompt.format_messages(topic=request.topic)
        # Instead of passing a dict
        output = llm.invoke(formatted_prompt[0].content)

        essay_text = output if isinstance(output, str) else str(output)
        return {"essay": essay_text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
