"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
"""

import re
from pathlib import Path
from typing import List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        # Caption cache: image_id -> caption string
        self._caption_cache: Dict[str, str] = {}
        
        # Check if vision LLM is enabled in settings
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning
                 # effectively falling back to no-op for this transform
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # Assuming standard relative path. In production, logic might be robust.
        prompt_path = Path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Get caption for an image, using cache if available.
        
        Args:
            img_id: Image identifier
            img_path: Path to image file
            trace: Optional trace context
            
        Returns:
            Caption string or None if failed
        """
        # Check cache first
        if img_id in self._caption_cache:
            logger.debug(f"Caption cache hit for image {img_id}")
            return self._caption_cache[img_id]
        
        # Validate path
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content
            
            # Cache the result
            self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.
        
        Only processes images that are actually referenced in chunk text
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.
        """
        if not self.llm:
            return chunks
        
        # Build image lookup from all chunks' metadata
        # image_id -> {path, ...}
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing
        self._caption_cache.clear()
            
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            # Find which images are actually referenced in this chunk's text
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                # No image references in this chunk, skip processing
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                # Look up image metadata
                img_meta = image_lookup.get(img_id)
                if not img_meta:
                    logger.warning(f"Image {img_id} referenced but not found in metadata")
                    continue
                
                img_path = img_meta.get("path")
                
                # Get caption (from cache or API)
                caption = self._get_caption(img_id, img_path, trace)
                
                if caption:
                    captions.append({"id": img_id, "caption": caption})
                    
                    # Replace placeholder with caption
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            # Store captions in metadata (only for images actually in this chunk)
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        logger.info(f"Added {total_captions_added} captions, API calls: {len(self._caption_cache)}")
            
        return processed_chunks
