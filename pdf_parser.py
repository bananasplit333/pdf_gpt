import PyPDF2

def parse_pdf_data(pdf_location):
    pdf_data = []
    
    # Open the PDF file
    with open(pdf_location, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        
        # Iterate over each page in the PDF
        for page in reader.pages:
            # Extract the text content from the page
            text = page.extract_text()
            
            # Add the extracted text to the list
            pdf_data.append(text)
    
    # Return the parsed PDF data as a list
    return pdf_data
