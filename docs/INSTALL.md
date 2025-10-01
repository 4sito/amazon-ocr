# Installation Instructions

This file covers installing system and Python dependencies on Linux
(Debian/Ubuntu), macOS, and Windows.

## A. System dependencies: Tesseract OCR & language data

**Why:** pytesseract is a wrapper—Tesseract binary must be installed separately.

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-ita
```

### Fedora / RHEL / CentOS

```bash
sudo dnf install -y tesseract
```

### macOS (Homebrew)

```zsh
brew update
brew install tesseract
```

Add language data if needed: brew install tesseract-lang (if available) or
download traineddata files

### Windows

1. Download installer (e.g. UB Mannheim build):
   [tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

2. Run installer and add Tesseract install dir to PATH (e.g.,
   `C:\Program Files\Tesseract-OCR`).

3. Add language data: place `ita.traineddata` into the `tessdata` folder inside
   the Tesseract install directory.

Verify:

```cmd
tesseract --version tesseract --list-langs
```

If pytesseract cannot find the tesseract binary on `Windows`, set:

```py
import pytesseract pytesseract.pytesseract.tesseract_cmd = r"C:\Program
Files\Tesseract-OCR\tesseract.exe"
```

## B. Python dependencies

1. Create & activate venv

- Linux/macOS:

```bash
python -m venv .venv 
source .venv/bin/activate
```

- Windows (cmd):

```cmd
python -m venv .venv .venv\Scripts\activate
```

2. Install Python packages, from repo root:

```bash
pip install --upgrade pip setuptools wheel pip install -r requirements.txt
```
