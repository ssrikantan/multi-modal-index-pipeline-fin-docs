import fitz  # PyMuPDF
import pandas as pd
import re
import os
from openai import AzureOpenAI
import base64
from mimetypes import guess_type
import config

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = config.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = config.AZURE_OPENAI_API_KEY
DEPLOYMENT_ID = config.DEPLOYMENT_ID
API_VERSION = config.API_VERSION
input_file_name = config.input_file_name
output_file_name = config.output_file_name

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=API_VERSION,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_ID}"
)



# Function to convert PDF pages to images
def convert_pdf_pages_to_images(pdf_path, zoom=5.0):
    pdf_document = fitz.open(pdf_path)
    image_data = {}

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # Zoom to improve quality
        image_path = f'images/page_{page_number + 1}.png'
        pix.save(image_path)
        image_data[page_number + 1] = [image_path]  # Store as list to be consistent with image analysis
        print("image path is ", image_path)
    return image_data

# Function to analyze images using Azure OpenAI GPT-4 API
def analyze_images_with_azure_gpt4(image_paths):
    summaries = []
    for image_path in image_paths:
        # Open the image file
        with open(image_path, 'rb') as image_file:
            image_data = None
            # print("image_file", image_file)
            # Guess the MIME type of the image based on the file extension
            mime_type, _ = guess_type(image_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'  # Default MIME type if none is found

            # Read and encode the image file
            with open(image_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:{mime_type};base64,{base64_encoded_data}"
            
            # Prompt for GPT-4 image analysis
            prompt = """
            **Role**: Finance Consultant specializing in analyzing financial statements and providing detailed insights.

            **Task**: Analyze the provided image from a financial statement (e.g., a graph, chart, or table) and deliver a comprehensive narration that goes beyond obvious observations.

            **Your response should include:**

            1. **Title or Heading**: Clearly state the title or heading of the graph, chart, or table.

            2. **Description**: Provide a detailed description of the image, explaining what it represents and the significance of its content.

            3. **Details**:
            - For graphs and charts:
                - Describe the axes labels and units.
                - Explain the legends and what each plotted value represents.
            - For tables:
                - Describe the rows and columns.
                - Highlight any significant data points or trends.

            4. **Insights**: Offer in-depth insights on the significance of the data presented, interpreting what it implies in the context of the overall document.

            5. **Page Reference**: Include the page number found in the image as a footnote, referring to it as 'Page in the Manual', so users can navigate to the corresponding page.

            *Footnote Example:*

            _Page in the Manual: 42_
            """

            response = client.chat.completions.create(
            model=DEPLOYMENT_ID,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": prompt 
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ] } 
            ],
            max_tokens=2000 
            )
            summaries.append(response.choices[0].message.content)
    return summaries

# Create markdown output
def create_output_document(image_data):
    with open(output_file_name, "w", encoding="utf-8") as f:
        # Write header
        # f.write("# PDF Parsing Output\n\n")

        # for page_number in range(1, max(text_data.keys() | image_data.keys()) + 1):
        for page_number in range(1,  max(image_data.keys()) + 1):
            # Write page number
            print(f"********************  Page {page_number} **************")
            # f.write(f"## Page {page_number}:\n\n")
            # f.flush()
            
            # Write image summaries and URLs
            if page_number in image_data:
                print("matched image data", image_data[page_number])
                image_summaries = analyze_images_with_azure_gpt4(image_data[page_number])
                for i, image_summary in enumerate(image_summaries):
                    print("image summary is ", image_summary)
                    # f.write(f"### Image {i+1} Summary:\n")
                    f.write(f"{image_summary}\n\n")
                    f.write(f"![Image {i+1}](file://{os.path.abspath(image_data[page_number][i])})\n\n")
                    f.flush()
        # Write footer
        f.write("\n---\nGenerated using PDF parser.\n")
        f.flush()
        f.close()

# Main function to parse the PDF
def parse_pdf(pdf_path):

    # Convert PDF pages to images
    image_data = convert_pdf_pages_to_images(pdf_path)
    
    # Create markdown output
    create_output_document(image_data)

# Specify the PDF path and run the parser
pdf_path = input_file_name  # Path to your PDF file
parse_pdf(pdf_path)
