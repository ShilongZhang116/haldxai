# -*- coding: utf-8 -*-
"""
Helper utility functions for HALDxAI platform.
"""

import re
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable
from functools import wraps
import requests
from email.utils import parseaddr


def get_project_root() -> Path:
    """Get the project root directory.
    
    Looks for .git directory, pyproject.toml, or setup.py to identify project root.
    
    Returns:
        Path to project root directory
    """
    current = Path.cwd()
    
    for _ in range(10):  # Limit search depth to prevent infinite loops
        if (current / ".git").exists() or \
           (current / "pyproject.toml").exists() or \
           (current / "setup.py").exists():
            return current
        current = current.parent
    
    # Fallback to current directory
    return Path.cwd()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def clean_text(text: str, 
              remove_extra_whitespace: bool = True,
              remove_special_chars: bool = False,
              lowercase: bool = False) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_extra_whitespace: Remove extra whitespace and newlines
        remove_special_chars: Remove special characters except basic punctuation
        lowercase: Convert text to lowercase
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters
    if remove_special_chars:
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    return text


def normalize_text(text: str) -> str:
    """Normalize text for matching and comparison.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Standardize unicode characters
    text = text.replace('α', 'alpha').replace('β', 'beta')
    
    # Standardize dashes and quotes
    text = re.sub(r'[\u2010-\u2015]', '-', text)  # Various dashes
    text = re.sub(r'[\u2018\u2019]', "'", text)    # Single quotes
    text = re.sub(r'[\u201c\u201d]', '"', text)    # Double quotes
    
    return text


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    try:
        name, addr = parseaddr(email)
        if not addr:
            return False
        
        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, addr) is not None
    except Exception:
        return False


def validate_api_key(api_key: str, min_length: int = 10) -> bool:
    """Validate API key format.
    
    Args:
        api_key: API key to validate
        min_length: Minimum required length
        
    Returns:
        True if API key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    return len(api_key.strip()) >= min_length


def retry_with_backoff(max_retries: int = 3, 
                     initial_delay: float = 1.0,
                     max_delay: float = 60.0,
                     backoff_factor: float = 2.0,
                     exceptions: tuple = (Exception,)):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.
    
    Args:
        data: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def flatten_dict(data: Dict[str, Any], 
                separator: str = '.', 
                prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary.
    
    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.
    
    Later dictionaries override earlier ones.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def generate_hash(data: Union[str, bytes, Dict[str, Any]]) -> str:
    """Generate SHA-256 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def download_file(url: str, 
                destination: Union[str, Path],
                timeout: int = 30,
                chunk_size: int = 8192) -> bool:
    """Download file from URL.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        timeout: Request timeout in seconds
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        
        return True
    except Exception:
        return False


def safe_filename(filename: str) -> str:
    """Make filename safe for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace unsafe characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    safe_chars = safe_chars.strip(' .')
    
    # Ensure not empty
    if not safe_chars:
        safe_chars = 'unnamed_file'
    
    # Limit length
    if len(safe_chars) > 255:
        name, ext = Path(safe_chars).stem, Path(safe_chars).suffix
        safe_chars = name[:255-len(ext)] + ext
    
    return safe_chars


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def timer(func):
    """Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def cache_to_file(cache_dir: Optional[Union[str, Path]] = None):
    """Decorator to cache function results to files.
    
    Args:
        cache_dir: Directory to store cache files
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_hash({
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            })
            
            # Set cache directory
            if cache_dir is None:
                cache_path = Path(".cache")
            else:
                cache_path = Path(cache_dir)
            
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = cache_path / f"{cache_key}.json"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_result = json.load(f)
                    return cached_result
                except Exception:
                    pass  # Cache is corrupted, continue with function execution
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, default=str)
            except Exception:
                pass  # Failed to cache, but function executed successfully
            
            return result
        
        return wrapper
    return decorator


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears valid, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def extract_pmid(text: str) -> Optional[str]:
    """Extract PubMed ID (PMID) from text.
    
    Args:
        text: Text to search for PMID
        
    Returns:
        PMID if found, None otherwise
    """
    # Common PMID patterns
    pmid_patterns = [
        r'PMID:\s*(\d+)',
        r'pmid:\s*(\d+)',
        r'PMID\s*[:=]\s*(\d+)',
        r'pmid\s*[:=]\s*(\d+)',
        r'\b(\d{8})\b',  # 8-digit numbers (common PMID format)
    ]
    
    for pattern in pmid_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def batch_process(items: List[Any], 
                process_func, 
                batch_size: int = 100,
                show_progress: bool = True) -> List[Any]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to process each batch
        batch_size: Size of each batch
        show_progress: Show progress indicator
        
    Returns:
        List of results from all batches
    """
    results = []
    batches = chunk_list(items, batch_size)
    total_batches = len(batches)
    
    for i, batch in enumerate(batches):
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        if show_progress:
            print(f"Processed batch {i + 1}/{total_batches} ({len(batch)} items)")
    
    return results