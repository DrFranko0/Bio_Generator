from langchain_community import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

def generate_bio_with_langchain(preferences):
    # Create the prompt template
    prompt = f"""
    Create a personalized bio based on the following preferences:
    Career: {preferences['career']}
    Personality: {preferences['personality']}
    Interests: {preferences['interests']}
    Relationship: {preferences['relationship']}
    """
    
    model = OllamaLLM(model="llama3.1")  # Using Ollama's Llama 3.1 model
    prompt_template = PromptTemplate(template=prompt, input_variables=[])
    runnable = prompt_template | model
    
    print(f"Generated Prompt: {prompt}")
    
    try:
        bio = runnable.invoke({})
        
        print(f"Generated Bio: {bio}")
        
    except Exception as e:
        print(f"Error generating bio: {e}")
        bio = "Error generating bio. Please try again later."
    
    return bio
