# product_extractor/cli.py
import argparse
from .extractor import extract_products_from_pdf

def main():
    parser = argparse.ArgumentParser(description="Extract product info from catalog PDF.")
    parser.add_argument('pdf', nargs='?', help='PDF file to process.')
    parser.add_argument('--output', default='output_products_data', help='Output directory root.')
    parser.add_argument('--min-chars', type=int, default=400, help='Minimum chars to treat as product block.')
    parser.add_argument('--x-threshold', type=int, default=400)
    parser.add_argument('--y-threshold', type=int, default=100)
    parser.add_argument('--min-confidence', type=int, default=30)
    args = parser.parse_args()

    if args.pdf:
        extract_products_from_pdf(
            args.pdf,
            output_dir=args.output,
            x_threshold=args.x_threshold,
            y_threshold=args.y_threshold,
            min_chars=args.min_chars,
            min_confidence=args.min_confidence
        )
    else:
        print("Please provide a PDF path. Example: python -m product_extractor.cli myfile.pdf")

if __name__ == "__main__":
    main()
