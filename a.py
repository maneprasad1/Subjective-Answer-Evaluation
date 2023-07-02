import difflib
import PyPDF2

# Function to compare the extracted text
def compare_text(pdf_text, notes_text):
    diff_ratio = difflib.SequenceMatcher(None, pdf_text, notes_text).ratio()
    return diff_ratio
 
# Read text from the file extracted by Google Lens
with open("C:\\Users\\HP\\Desktop\\Untitled document.txt", 'r', encoding='utf-8') as file:
    text_content = file.read()
    # words = notes_text.split()

    # # Iterate over each word
    # for word in words:
    #     print(word)
# print(notes_text)
    # Split the text content into words
    words = text_content.split()
    notes_text=""
    # Iterate over each word
    for word in words:
        notes_text += word
print(notes_text)

# Read text from the PDF file

with open("C:\\Users\\HP\\Desktop\\Untitled document.pdf", 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
print(pdf_text)

# Compare the extracted text
similarity_ratio = compare_text(pdf_text, notes_text)
# Print the comparison result
print(f"Similarity Ratio: {similarity_ratio}")
if similarity_ratio >= 0.3:
    print("The handwritten notes and the PDF content are identical.")
else:
    print("The handwritten notes and the PDF content are not identical.")
