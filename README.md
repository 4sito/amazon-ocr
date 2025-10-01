# Product Extractor

Lightweight pipeline to extract product blocks (images + text fields) from
catalog PDFs using PyMuPDF, OpenCV and Tesseract OCR.\
Splits responsibilities into modules: PDF rendering → OCR → region detection →
product analysis → Excel export.

## Features

- Render PDF pages to images (PyMuPDF)
- Detect colored/visual product regions (OpenCV)
- OCR (Tesseract / pytesseract) with language support
- Heuristics to split rows/columns and extract price/rating/reviews/delivery
- Saves product images, debug images, and an Excel sheet with extracted fields
- CLI + programmatic API
