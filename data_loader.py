# data_loader.py
"""
Dataset Loading and Management for Nepal AI Chatbot
Handles loading, validation, and preprocessing of training data
"""

import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Advanced dataset loader with validation and preprocessing
    Supports multiple dataset formats and provides data quality metrics
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or Config.DATASET_PATH
        self.raw_data = None
        self.processed_data = []
        self.data_stats = {}
        self.validation_errors = []
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load and validate the dataset
        
        Returns:
            List of validated dataset entries
        """
        try:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            
            # Load raw data
            self._load_raw_data()
            
            # Process and validate data
            self._process_data()
            
            # Generate statistics
            self._generate_statistics()
            
            # Validate data quality
            self._validate_data_quality()
            
            logger.info(f"Successfully loaded {len(self.processed_data)} examples")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _load_raw_data(self):
        """Load raw data from JSON file"""
        dataset_file = Path(self.dataset_path)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        if not dataset_file.suffix.lower() == '.json':
            raise ValueError(f"Unsupported file format: {dataset_file.suffix}")
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error: {str(e)}")
    
    def _process_data(self):
        """Process raw data into standardized format"""
        if isinstance(self.raw_data, list):
            # Direct list format
            self.processed_data = self._process_list_format(self.raw_data)
        
        elif isinstance(self.raw_data, dict):
            # Dictionary format - try common keys
            for key in ['phrases', 'data', 'examples', 'training_data', 'dataset']:
                if key in self.raw_data:
                    if isinstance(self.raw_data[key], list):
                        self.processed_data = self._process_list_format(self.raw_data[key])
                        break
            
            if not self.processed_data:
                # Try to process as single-level dict
                self.processed_data = [self.raw_data]
        
        else:
            raise ValueError(f"Unsupported data format: {type(self.raw_data)}")
        
        if not self.processed_data:
            raise ValueError("No valid data entries found in dataset")
    
    def _process_list_format(self, data_list: List) -> List[Dict]:
        """Process list format data"""
        processed_entries = []
        
        for i, entry in enumerate(data_list):
            if not isinstance(entry, dict):
                self.validation_errors.append(f"Entry {i}: Not a dictionary")
                continue
            
            # Validate required fields
            if not self._validate_entry(entry, i):
                continue
            
            # Process and clean the entry
            processed_entry = self._clean_entry(entry)
            processed_entries.append(processed_entry)
        
        return processed_entries
    
    def _validate_entry(self, entry: Dict, index: int) -> bool:
        """Validate a single data entry"""
        required_fields = ['input']  # Minimum required field
        
        for field in required_fields:
            if field not in entry or not entry[field]:
                self.validation_errors.append(f"Entry {index}: Missing required field '{field}'")
                return False
        
        # Check for reasonable content length
        if len(entry['input']) < 3:
            self.validation_errors.append(f"Entry {index}: Input too short")
            return False
        
        if len(entry['input']) > 1000:
            self.validation_errors.append(f"Entry {index}: Input too long (>1000 chars)")
            return False
        
        # Validate output if present
        if 'output' in entry and entry['output']:
            if len(entry['output']) < 10:
                self.validation_errors.append(f"Entry {index}: Output too short")
                return False
        
        return True
    
    def _clean_entry(self, entry: Dict) -> Dict:
        """Clean and standardize a data entry"""
        cleaned_entry = {}
        
        # Standard field mapping
        field_mapping = {
            'input': ['input', 'question', 'query', 'text'],
            'output': ['output', 'answer', 'response', 'reply'],
            'nepali': ['nepali', 'nepal', 'nepali_translation', 'translation']
        }
        
        for standard_field, possible_fields in field_mapping.items():
            for field in possible_fields:
                if field in entry and entry[field]:
                    cleaned_entry[standard_field] = str(entry[field]).strip()
                    break
        
        # Copy any additional fields
        additional_fields = ['category', 'type', 'id', 'similarity', 'metadata']
        for field in additional_fields:
            if field in entry:
                cleaned_entry[field] = entry[field]
        
        # Add processing metadata
        cleaned_entry['processed_at'] = __import__('datetime').datetime.now().isoformat()
        
        return cleaned_entry
    
    def _generate_statistics(self):
        """Generate dataset statistics"""
        if not self.processed_data:
            return
        
        self.data_stats = {
            'total_entries': len(self.processed_data),
            'validation_errors': len(self.validation_errors),
            'fields_present': self._analyze_field_presence(),
            'content_stats': self._analyze_content_stats(),
            'language_distribution': self._analyze_language_distribution()
        }
    
    def _analyze_field_presence(self) -> Dict[str, int]:
        """Analyze which fields are present in the dataset"""
        field_counts = {}
        
        for entry in self.processed_data:
            for field in entry.keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        
        return field_counts
    
    def _analyze_content_stats(self) -> Dict[str, Any]:
        """Analyze content statistics"""
        input_lengths = [len(entry['input']) for entry in self.processed_data]
        output_lengths = [len(entry.get('output', '')) for entry in self.processed_data if entry.get('output')]
        
        stats = {
            'input_length': {
                'mean': sum(input_lengths) / len(input_lengths),
                'min': min(input_lengths),
                'max': max(input_lengths),
                'median': sorted(input_lengths)[len(input_lengths)//2]
            }
        }
        
        if output_lengths:
            stats['output_length'] = {
                'mean': sum(output_lengths) / len(output_lengths),
                'min': min(output_lengths),
                'max': max(output_lengths),
                'median': sorted(output_lengths)[len(output_lengths)//2]
            }
        
        return stats
    
    def _analyze_language_distribution(self) -> Dict[str, int]:
        """Analyze language distribution in the dataset"""
        distribution = {
            'has_nepali': 0,
            'english_only': 0,
            'bilingual': 0
        }
        
        for entry in self.processed_data:
            has_nepali = bool(entry.get('nepali'))
            has_english = bool(entry.get('output'))
            
            if has_nepali:
                distribution['has_nepali'] += 1
            if has_english and has_nepali:
                distribution['bilingual'] += 1
            elif has_english:
                distribution['english_only'] += 1
        
        return distribution
    
    def _validate_data_quality(self):
        """Validate overall data quality"""
        total_entries = len(self.processed_data)
        error_count = len(self.validation_errors)
        
        if error_count > 0:
            logger.warning(f"Found {error_count} validation errors in dataset")
            for error in self.validation_errors[:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
            
            if error_count > 10:
                logger.warning(f"  ... and {error_count - 10} more errors")
        
        # Quality metrics
        quality_score = (total_entries - error_count) / total_entries if total_entries > 0 else 0
        
        if quality_score < 0.8:
            logger.warning(f"Dataset quality score is low: {quality_score:.2f}")
        else:
            logger.info(f"Dataset quality score: {quality_score:.2f}")
        
        self.data_stats['quality_score'] = quality_score
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        return {
            'file_path': self.dataset_path,
            'statistics': self.data_stats,
            'validation_errors': self.validation_errors,
            'sample_entries': self.processed_data[:3] if self.processed_data else []
        }
    
    def export_cleaned_dataset(self, output_path: str):
        """Export the cleaned and processed dataset"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'source': self.dataset_path,
                        'processed_at': __import__('datetime').datetime.now().isoformat(),
                        'statistics': self.data_stats
                    },
                    'data': self.processed_data
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported cleaned dataset to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dataset: {str(e)}")
            raise
    
    def get_categories(self) -> List[str]:
        """Get unique categories from the dataset"""
        categories = set()
        for entry in self.processed_data:
            if 'category' in entry and entry['category']:
                categories.add(entry['category'])
        return sorted(list(categories))
    
    def filter_by_category(self, category: str) -> List[Dict]:
        """Filter dataset by category"""
        return [
            entry for entry in self.processed_data 
            if entry.get('category', '').lower() == category.lower()
        ]
    
    def search_dataset(self, query: str, field: str = 'input') -> List[Dict]:
        """Search dataset for entries containing query"""
        query_lower = query.lower()
        results = []
        
        for entry in self.processed_data:
            if field in entry and query_lower in entry[field].lower():
                results.append(entry)
        
        return results