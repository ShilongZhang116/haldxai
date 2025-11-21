# -*- coding: utf-8 -*-
"""
Input/Output utilities for HALDxAI platform.
"""

import json
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
import pandas as pd
import yaml
import hashlib

def load_json(file_path: Union[str, Path], 
             encoding: str = 'utf-8',
             raise_on_error: bool = True) -> Optional[Dict[str, Any]]:
    """Load JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Parsed JSON data or None if failed
    """
    try:
        path = Path(file_path)
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        if raise_on_error:
            raise
        return None


def save_json(data: Any, 
             file_path: Union[str, Path],
             encoding: str = 'utf-8',
             indent: int = 2,
             ensure_dir: bool = True) -> bool:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        encoding: File encoding
        indent: JSON indentation
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception:
        return False


def load_csv(file_path: Union[str, Path],
             encoding: str = 'utf-8',
             delimiter: str = ',',
             **kwargs) -> Optional[pd.DataFrame]:
    """Load CSV file as pandas DataFrame.
    
    Args:
        file_path: Path to CSV file
        encoding: File encoding
        delimiter: CSV delimiter
        **kwargs: Additional arguments for pandas.read_csv
        
    Returns:
        DataFrame or None if failed
    """
    try:
        path = Path(file_path)
        return pd.read_csv(path, encoding=encoding, delimiter=delimiter, **kwargs)
    except Exception:
        return None


def save_csv(data: Union[pd.DataFrame, List[Dict[str, Any]], List[List[Any]]],
             file_path: Union[str, Path],
             encoding: str = 'utf-8',
             delimiter: str = ',',
             index: bool = False,
             ensure_dir: bool = True,
             **kwargs) -> bool:
    """Save data to CSV file.
    
    Args:
        data: Data to save (DataFrame, list of dicts, or list of lists)
        file_path: Output file path
        encoding: File encoding
        delimiter: CSV delimiter
        index: Whether to write index (for DataFrame)
        ensure_dir: Create parent directories if needed
        **kwargs: Additional arguments for pandas.DataFrame.to_csv
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list) and data and isinstance(data[0], list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            df = data
        
        df.to_csv(path, encoding=encoding, sep=delimiter, index=index, **kwargs)
        return True
    except Exception:
        return False


def load_yaml(file_path: Union[str, Path],
              encoding: str = 'utf-8',
              raise_on_error: bool = True) -> Optional[Dict[str, Any]]:
    """Load YAML file.
    
    Args:
        file_path: Path to YAML file
        encoding: File encoding
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Parsed YAML data or None if failed
    """
    try:
        path = Path(file_path)
        with open(path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    except Exception as e:
        if raise_on_error:
            raise
        return None


def save_yaml(data: Any,
             file_path: Union[str, Path],
             encoding: str = 'utf-8',
             ensure_dir: bool = True) -> bool:
    """Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
        encoding: File encoding
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception:
        return False


def load_pickle(file_path: Union[str, Path],
               raise_on_error: bool = True) -> Optional[Any]:
    """Load pickle file.
    
    Args:
        file_path: Path to pickle file
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Unpickled data or None if failed
    """
    try:
        path = Path(file_path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if raise_on_error:
            raise
        return None


def save_pickle(data: Any,
              file_path: Union[str, Path],
              ensure_dir: bool = True) -> bool:
    """Save data to pickle file.
    
    Args:
        data: Data to save
        file_path: Output file path
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False


def read_text(file_path: Union[str, Path],
             encoding: str = 'utf-8',
             raise_on_error: bool = True) -> Optional[str]:
    """Read text file.
    
    Args:
        file_path: Path to text file
        encoding: File encoding
        raise_on_error: Whether to raise exception on error
        
    Returns:
        File content or None if failed
    """
    try:
        path = Path(file_path)
        return path.read_text(encoding=encoding)
    except Exception as e:
        if raise_on_error:
            raise
        return None


def write_text(content: str,
              file_path: Union[str, Path],
              encoding: str = 'utf-8',
              ensure_dir: bool = True) -> bool:
    """Write text to file.
    
    Args:
        content: Text content to write
        file_path: Output file path
        encoding: File encoding
        ensure_dir: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False


def list_files(directory: Union[str, Path],
              pattern: str = "*",
              recursive: bool = False) -> List[Path]:
    """List files in directory matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    path = Path(directory)
    if recursive:
        return list(path.rglob(pattern))
    else:
        return list(path.glob(pattern))


def file_exists(file_path: Union[str, Path]) -> bool:
    """Check if file exists.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).is_file()


def directory_exists(dir_path: Union[str, Path]) -> bool:
    """Check if directory exists.
    
    Args:
        dir_path: Path to check
        
    Returns:
        True if directory exists, False otherwise
    """
    return Path(dir_path).is_dir()


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def copy_file(source: Union[str, Path],
             destination: Union[str, Path],
             overwrite: bool = False) -> bool:
    """Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src = Path(source)
        dst = Path(destination)
        
        if dst.exists() and not overwrite:
            return False
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def move_file(source: Union[str, Path],
             destination: Union[str, Path],
             overwrite: bool = False) -> bool:
    """Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src = Path(source)
        dst = Path(destination)
        
        if dst.exists() and not overwrite:
            return False
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        return False


def delete_file(file_path: Union[str, Path], 
              missing_ok: bool = True) -> bool:
    """Delete file.
    
    Args:
        file_path: Path to file to delete
        missing_ok: Whether to ignore if file doesn't exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        path.unlink(missing_ok=missing_ok)
        return True
    except Exception:
        return False


def create_directory(dir_path: Union[str, Path],
                  exist_ok: bool = True,
                  parents: bool = True) -> bool:
    """Create directory.
    
    Args:
        dir_path: Directory path to create
        exist_ok: Whether to ignore if directory already exists
        parents: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=parents, exist_ok=exist_ok)
        return True
    except Exception:
        return False


def iterate_csv_rows(file_path: Union[str, Path],
                  encoding: str = 'utf-8',
                  delimiter: str = ',',
                  chunk_size: Optional[int] = None,
                  **kwargs) -> Iterator[List[Dict[str, Any]]]:
    """Iterate over CSV file in chunks.
    
    Args:
        file_path: Path to CSV file
        encoding: File encoding
        delimiter: CSV delimiter
        chunk_size: Number of rows per chunk (None for all at once)
        **kwargs: Additional arguments for pandas.read_csv
        
    Yields:
        List of row dictionaries for each chunk
    """
    path = Path(file_path)
    
    if chunk_size is None:
        # Read entire file
        df = pd.read_csv(path, encoding=encoding, delimiter=delimiter, **kwargs)
        yield df.to_dict('records')
    else:
        # Read in chunks
        for chunk in pd.read_csv(path, encoding=encoding, delimiter=delimiter, 
                             chunksize=chunk_size, **kwargs):
            yield chunk.to_dict('records')


def backup_file(file_path: Union[str, Path],
               backup_dir: Optional[Union[str, Path]] = None,
               timestamp: bool = True) -> Optional[Path]:
    """Create backup of file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backup files (default: same as file)
        timestamp: Whether to add timestamp to backup name
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        src = Path(file_path)
        
        if backup_dir is None:
            backup_dir = src.parent
        else:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        if timestamp:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{src.stem}_{timestamp_str}{src.suffix}"
        else:
            backup_name = f"{src.stem}_backup{src.suffix}"
        
        backup_path = backup_dir / backup_name
        
        # Copy file
        import shutil
        shutil.copy2(src, backup_path)
        
        return backup_path
    except Exception:
        return None


def get_file_hash(file_path: Union[str, Path],
                 algorithm: str = 'md5') -> Optional[str]:
    """Calculate hash of file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string or None if failed
    """
    try:
        path = Path(file_path)
        
        if algorithm.lower() == 'md5':
            hasher = hashlib.md5()
        elif algorithm.lower() == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm.lower() == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    except Exception:
        return None