from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Read the content from the existing file
with open('thesis_report.docx', 'r') as f:
    content = f.read()

# Create a new document
document = Document()

# Set the document properties
sections = document.sections
for section in sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Set the font and line spacing
style = document.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
paragraph_format = style.paragraph_format
paragraph_format.line_spacing = 1.5

# Add the content to the document
document.add_paragraph(content)

# Save the document
document.save('formatted_thesis_report.docx')
