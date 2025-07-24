from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
load_dotenv()
# Configuration
#azure ai foundry endpoint
endpoint = "https://ai-demo-foundry-deva.cognitiveservices.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")  # Replace with your actual key
OPEN_AI_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")  # Replace with your actual endpoint
# Create client
client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(AZURE_API_KEY)
)

def analyze_document(file_path, model_id="prebuilt-layout"):
    """Simple document analysis"""
    try:
        with open(file_path, "rb") as document:
            # Analyze document
            poller = client.begin_analyze_document(
                model_id=model_id,
                content_type="application/octet-stream",
                body=document
            )
            
            result = poller.result()
            
            # Print extracted text
            print("=== EXTRACTED TEXT ===")
            print(result.content)
            
            # Print tables if found
            if result.tables:
                print(f"\n=== TABLES FOUND: {len(result.tables)} ===")
                for i, table in enumerate(result.tables):
                    print(f"Table {i+1}: {table.row_count} rows, {table.column_count} columns")
            
            return result
            
    except Exception as e:
        print(f"Error: {e}")
        return None
def analyze_image_foundry(image_path: str):
    client = ImageAnalysisClient(
        endpoint="https://ai-demo-foundry-deva.cognitiveservices.azure.com/",
        credential=AzureKeyCredential(AZURE_API_KEY)
    )
    #we give features here

    features =  [VisualFeatures.TAGS, VisualFeatures.PEOPLE]

    with open(image_path, "rb") as f:
        result = client.analyze(
            image_data=f.read(),
            visual_features=features,
            gender_neutral_caption=True
        )

    caption = result.caption.text if result.caption else "No caption"
    print("tags", result.tags)
    people_count = len(result.people or [])

    # print("Caption:", caption)
    # print("Tags:", tags)
    # print("People detected:", people_count)
    return {"caption": caption,  "people_count": people_count}

def gpt_check():
    client = AzureOpenAI(
        azure_endpoint=OPEN_AI_ENDPOINT, #when deployed it gives url place it here
        api_key=AZURE_API_KEY, #foundry proect key
        api_version="2024-02-01"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",  # Use your GPT-4.1 deployment name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, GPT-4.1! What can you do?"}
        ]
    )
    return response.choices[0].message.content
# Example usage
if __name__ == "__main__":
    # Replace with your document path
    document_path = "imp_docs/dummy.png"
    
    # Basic analysis (extracts text, tables, layout)
    # result = analyze_document(document_path)
    # result = analyze_image_foundry(document_path)
    result = gpt_check()
    print(result)
    # For specific document types, use different model_id:
    # analyze_document(document_path, "prebuilt-invoice")     # For invoices
    # analyze_document(document_path, "prebuilt-receipt")     # For receipts
    # analyze_document(document_path, "prebuilt-idDocument")  # For ID cards