import fitz  # PyMuPDF
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image
import io


class PDFToPNGConverter:
    """
    PDF to PNG converter with image size control and PIL Image export capability.
    Resizes images so the longest side is a specified maximum length.
    """
    
    def __init__(self, max_long_side: int = 2000):
        """
        Initialize the converter.
        
        Args:
            max_long_side: Maximum length of the longest side of output images in pixels.
        """
        self.max_long_side = max_long_side
        self.supported_extensions = {'.pdf'}
    
    def _validate_input_file(self, input_path: Union[str, Path]) -> Path:
        """
        Validate the input file exists and has a supported format.
        
        Args:
            input_path: Path to the input file.
            
        Returns:
            Validated Path object.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(input_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: {self.supported_extensions}")
        
        return path
    
    def _create_output_dir(self, output_path: Union[str, Path]) -> Path:
        """
        Create the output directory if it doesn't exist.
        
        Args:
            output_path: Path to output file or directory.
            
        Returns:
            Path object for the output directory.
        """
        output_dir = Path(output_path)
        
        # If output_path is a file path (has .png extension), use its parent directory
        if output_path.suffix.lower() == '.png':
            output_dir = output_dir.parent
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _calculate_scale_factor(self, width: int, height: int) -> float:
        """
        Calculate the scale factor to resize the image so the longest side equals max_long_side.
        
        Args:
            width: Original width in pixels.
            height: Original height in pixels.
            
        Returns:
            Scale factor for resizing (1.0 means no scaling needed).
        """
        long_side = max(width, height)
        
        # If image is already smaller than max_long_side, don't scale up
        if long_side <= self.max_long_side:
            return 1.0
        
        # Calculate scale factor to make long side equal to max_long_side
        return self.max_long_side / long_side
    
    def _convert_page_to_png(self, page: fitz.Page, scale_factor: float, 
                            output_path: Path, page_num: int) -> str:
        """
        Convert a single PDF page to PNG image and save to file.
        
        Args:
            page: PDF page object.
            scale_factor: Scale factor for resizing.
            output_path: Output path (file or directory).
            page_num: Page number (0-indexed).
            
        Returns:
            Path to the saved PNG file.
        """
        # Create transformation matrix for scaling
        mat = fitz.Matrix(scale_factor, scale_factor)
        
        # Get pixmap of the page with the specified scaling
        # alpha=False ensures no transparency for PNG compatibility
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Generate output filename based on output_path type
        if output_path.is_dir():
            # If output_path is a directory, create file inside it
            output_filename = output_path / f"{output_path.stem}_page_{page_num + 1:03d}.png"
        else:
            # If output_path is a file path
            if page_num > 0:
                # For multi-page PDFs, add page number suffix
                stem = output_path.stem
                suffix = f"_page_{page_num + 1:03d}"
                output_filename = output_path.parent / f"{stem}{suffix}.png"
            else:
                # Single page PDF
                output_filename = output_path
        
        # Save as PNG format
        pix.save(str(output_filename))
        
        return str(output_filename)
    
    def convert_page_to_pil_image(self, page: fitz.Page, scale_factor: float) -> Image.Image:
        """
        Convert a PDF page to PIL Image object.
        
        Args:
            page: PDF page object.
            scale_factor: Scale factor for resizing.
            
        Returns:
            PIL Image object.
        """
        # Create transformation matrix for scaling
        mat = fitz.Matrix(scale_factor, scale_factor)
        
        # Get pixmap of the page with the specified scaling
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to bytes
        img_bytes = pix.tobytes("ppm")
        
        # Create PIL Image from bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        return img
    
    def get_page_as_pil(self, input_path: Union[str, Path], 
                       page_num: int = 0) -> Image.Image:
        """
        Get a specific page from PDF as PIL Image object.
        
        Args:
            input_path: Path to the input PDF file.
            page_num: Page number to extract (0-indexed).
            
        Returns:
            PIL Image object.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If input file format is not supported or page number is invalid.
            RuntimeError: If conversion fails.
        """
        # Validate input file
        input_path_obj = self._validate_input_file(input_path)
        
        try:
            # Open PDF document
            pdf_document = fitz.open(str(input_path_obj))
            
            # Validate page number
            if page_num < 0 or page_num >= len(pdf_document):
                pdf_document.close()
                raise ValueError(f"Page number {page_num} is out of range. "
                               f"Document has {len(pdf_document)} pages.")
            
            # Get the specified page
            page = pdf_document[page_num]
            
            # Get page dimensions
            width, height = page.rect.width, page.rect.height
            
            # Calculate scale factor for resizing
            scale_factor = self._calculate_scale_factor(width, height)
            
            # Convert page to PIL Image
            pil_image = self.convert_page_to_pil_image(page, scale_factor)
            
            # Close the document
            pdf_document.close()
            
            return pil_image
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to extract page {page_num} from {input_path_obj}: {e}")
    
    def get_multiple_pages_as_pil(self, input_path: Union[str, Path],
                                 page_nums: Optional[List[int]] = None) -> List[Image.Image]:
        """
        Get multiple pages from PDF as PIL Image objects.
        
        Args:
            input_path: Path to the input PDF file.
            page_nums: List of page numbers to extract (0-indexed).
                      If None, extracts all pages.
            
        Returns:
            List of PIL Image objects.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If input file format is not supported or page numbers are invalid.
            RuntimeError: If conversion fails.
        """
        # Validate input file
        input_path_obj = self._validate_input_file(input_path)
        
        try:
            # Open PDF document
            pdf_document = fitz.open(str(input_path_obj))
            total_pages = len(pdf_document)
            
            # If no specific pages provided, extract all pages
            if page_nums is None:
                page_nums = list(range(total_pages))
            
            # Validate all page numbers
            for page_num in page_nums:
                if page_num < 0 or page_num >= total_pages:
                    pdf_document.close()
                    raise ValueError(f"Page number {page_num} is out of range. "
                                   f"Document has {total_pages} pages.")
            
            results = []
            
            # Process each requested page
            for page_num in page_nums:
                page = pdf_document[page_num]
                
                # Get page dimensions
                width, height = page.rect.width, page.rect.height
                
                # Calculate scale factor for resizing
                scale_factor = self._calculate_scale_factor(width, height)
                
                # Convert page to PIL Image
                pil_image = self.convert_page_to_pil_image(page, scale_factor)
                
                results.append(pil_image)
                
            # Close the document
            pdf_document.close()
            
            return results
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to extract pages from {input_path_obj}: {e}")
    
    def convert(self, input_path: Union[str, Path], 
               output_path: Optional[Union[str, Path]] = None) -> List[str]:
        """
        Convert PDF file to PNG images and save to files.
        
        Args:
            input_path: Path to the input PDF file.
            output_path: Path for output PNG file or directory. 
                        If it's a directory, files will be created inside it.
                        If None, a directory with PDF filename will be created.
                        
        Returns:
            List of paths to created PNG files.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If input file format is not supported.
            RuntimeError: If conversion fails.
        """
        # Validate input file
        input_path_obj = self._validate_input_file(input_path)
        
        # Set default output path if not provided
        if output_path is None:
            # Create directory with PDF filename + '_png' suffix
            output_path = input_path_obj.parent / f"{input_path_obj.stem}_png"
        
        output_path_obj = Path(output_path)
        
        # Create output directory
        self._create_output_dir(output_path_obj)
        
        saved_files = []
        
        try:
            # Open PDF document
            pdf_document = fitz.open(str(input_path_obj))
            num_pages = len(pdf_document)
            print(f"Processing: {input_path_obj.name} ({num_pages} pages)")
            
            # Process each page
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                
                # Get page dimensions
                width, height = page.rect.width, page.rect.height
                
                # Calculate scale factor for resizing
                scale_factor = self._calculate_scale_factor(width, height)
                
                # Convert page to PNG and save to file
                saved_file = self._convert_page_to_png(page, scale_factor, 
                                                      output_path_obj, page_num)
                saved_files.append(saved_file)
                
            # Close the document
            pdf_document.close()
            
            return saved_files
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Conversion failed for {input_path_obj}: {e}")
    
    def convert_batch(self, input_dir: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     recursive: bool = False) -> List[str]:
        """
        Convert all PDF files in a directory to PNG files.
        
        Args:
            input_dir: Directory containing PDF files.
            output_dir: Output directory for PNG files.
                       If None, a 'converted_png' directory will be created.
            recursive: If True, process PDF files in subdirectories recursively.
            
        Returns:
            List of paths to all created PNG files.
            
        Raises:
            FileNotFoundError: If input directory doesn't exist.
        """
        input_dir_obj = Path(input_dir)
        
        if not input_dir_obj.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir_obj}")
        
        # Set default output directory
        if output_dir is None:
            output_dir = input_dir_obj / "converted_png"
        
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # Build file search pattern
        pattern = "**/*.pdf" if recursive else "*.pdf"
        
        # Find all PDF files
        pdf_files = list(input_dir_obj.glob(pattern))
        
        if not pdf_files:
            return []
        
        all_saved_files = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                # Preserve relative directory structure
                relative_path = pdf_file.relative_to(input_dir_obj)
                file_output_dir = output_dir_obj / relative_path.parent / pdf_file.stem
                
                # Convert this PDF file
                saved_files = self.convert(pdf_file, file_output_dir)
                all_saved_files.extend(saved_files)
                
            except Exception as e:
                raise RuntimeError(f"Failed to process {pdf_file}: {e}")
        
        return all_saved_files


def main():
    """
    Main function demonstrating usage of the PDFToPNGConverter.
    """
    # Create converter instance with max long side = 2000 pixels
    converter = PDFToPNGConverter(max_long_side=2000)
    
    # Example 1: Convert single file with default output
    try:
        # Replace with your PDF file path
        pdf_file = "example.pdf"
        pdf_path = Path(pdf_file)
        
        if pdf_path.exists():
            # Convert with automatic output directory creation
            saved_files = converter.convert(pdf_file)
            
            print(f"\nOutput files:")
            for file_path in saved_files:
                print(f"  - {file_path}")
            
        else:
            print(f"Example file {pdf_file} not found.")
            print("Please provide your own PDF file path.")
            
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Format error: {e}")
    except RuntimeError as e:
        print(f"Conversion error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Extract specific page as PIL Image
    print("\n" + "=" * 50)
    print("Example 2: Extract page as PIL Image")
    print("=" * 50)
    
    try:
        if pdf_path.exists():
            # Extract first page as PIL Image
            pil_image, dimensions = converter.get_page_as_pil(pdf_file, page_num=0)
            width, height = dimensions
            
            print(f"Extracted page 1 as PIL Image:")
            print(f"  Image size: {width}×{height} pixels")
            print(f"  Image mode: {pil_image.mode}")
            print(f"  Image format: {pil_image.format if hasattr(pil_image, 'format') else 'N/A'}")
            
            # Example: Process the PIL Image
            # pil_image.show()  # Display the image
            # pil_image.save("extracted_page.png")  # Save to file
            
            # Example 3: Extract multiple pages
            print("\n" + "-" * 30)
            print("Example 3: Extract multiple pages as PIL Images")
            print("-" * 30)
            
            # Extract pages 0, 1, and 2 (first three pages)
            pages = converter.get_multiple_pages_as_pil(pdf_file, page_nums=[0, 1, 2])
            
            for i, (img, dims) in enumerate(pages):
                print(f"  Page {i+1}: {dims[0]}×{dims[1]} pixels")
            
            # Example 4: Extract all pages
            print("\n" + "-" * 30)
            print("Example 4: Extract all pages as PIL Images")
            print("-" * 30)
            
            all_pages = converter.get_multiple_pages_as_pil(pdf_file)
            print(f"Extracted all {len(all_pages)} pages as PIL Images")
            
        else:
            print("Skipping PIL extraction examples - PDF file not found.")
            
    except Exception as e:
        print(f"Error in PIL extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()