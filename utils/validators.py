"""
MÓDULO: SISTEMA DE VALIDACIONES
==============================
Centraliza todas las validaciones del sistema siguiendo el principio de responsabilidad única
"""

import os
import re
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .error_handler import ValidationError, ErrorSeverity

class ValidationResult:
    """Resultado de una validación"""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, warnings: List[str] = None, context: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.context = context or {}
    
    def add_error(self, error: str, field: str = ""):
        """Agrega un error"""
        self.is_valid = False
        error_msg = f"{field}: {error}" if field else error
        self.errors.append(error_msg)
    
    def add_warning(self, warning: str, field: str = ""):
        """Agrega una advertencia"""
        warning_msg = f"{field}: {warning}" if field else warning
        self.warnings.append(warning_msg)
    
    def merge(self, other: 'ValidationResult'):
        """Combina con otro resultado de validación"""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.context.update(other.context)
    
    def get_summary(self) -> str:
        """Obtiene resumen de la validación"""
        if self.is_valid:
            summary = "✅ Validación exitosa"
            if self.warnings:
                summary += f" ({len(self.warnings)} advertencias)"
        else:
            summary = f"❌ Validación falló ({len(self.errors)} errores"
            if self.warnings:
                summary += f", {len(self.warnings)} advertencias"
            summary += ")"
        
        return summary

class IValidator(ABC):
    """Interface base para todos los validadores"""
    
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Valida un valor"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna las reglas de validación"""
        pass

class URLValidator(IValidator):
    """Validador de URLs de video"""
    
    def __init__(self):
        self.supported_domains = [
            'youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com',
            'vimeo.com', 'www.vimeo.com',
            'dailymotion.com', 'www.dailymotion.com'
        ]
        
        self.youtube_patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
        ]
    
    def validate(self, url: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Valida URL de video"""
        result = ValidationResult()
        
        if not url or not isinstance(url, str):
            result.add_error("URL es requerida y debe ser texto", "url")
            return result
        
        url = url.strip()
        
        # Validar formato básico de URL
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                result.add_error("URL no tiene formato válido", "url")
                return result
        except Exception:
            result.add_error("URL malformada", "url")
            return result
        
        # Validar esquema
        if parsed.scheme not in ['http', 'https']:
            result.add_error("URL debe usar HTTP o HTTPS", "url")
        
        # Validar dominio soportado
        domain = parsed.netloc.lower()
        if not any(supported in domain for supported in self.supported_domains):
            result.add_warning(f"Dominio no reconocido: {domain}", "url")
            result.context['unsupported_domain'] = True
        
        # Validación específica para YouTube
        if 'youtube' in domain or 'youtu.be' in domain:
            if not self._validate_youtube_url(url):
                result.add_error("URL de YouTube no válida", "url")
            else:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    result.context['video_id'] = video_id
                    result.context['platform'] = 'youtube'
        
        # Validaciones adicionales
        if len(url) > 2000:
            result.add_warning("URL muy larga", "url")
        
        return result
    
    def _validate_youtube_url(self, url: str) -> bool:
        """Valida específicamente URLs de YouTube"""
        return any(re.search(pattern, url) for pattern in self.youtube_patterns)
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extrae ID de video de YouTube"""
        for pattern in self.youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "required": "URL es obligatoria",
            "format": "Debe ser una URL válida (http/https)",
            "domains": f"Dominios soportados: {', '.join(self.supported_domains)}",
            "youtube": "URLs de YouTube deben contener ID de video válido"
        }

class FileValidator(IValidator):
    """Validador de archivos"""
    
    def __init__(self):
        self.video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv']
        self.audio_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']
        self.text_extensions = ['.txt', '.srt', '.vtt', '.json']
        
        self.max_file_size_gb = 5.0  # 5GB máximo por defecto
        self.min_file_size_kb = 1.0  # 1KB mínimo
    
    def validate(self, file_path: Union[str, Path], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """Valida archivo"""
        result = ValidationResult()
        context = context or {}
        
        if not file_path:
            result.add_error("Ruta de archivo es requerida", "file_path")
            return result
        
        file_path = Path(file_path)
        
        # Validar existencia
        if not file_path.exists():
            result.add_error(f"Archivo no existe: {file_path}", "existence")
            return result
        
        # Validar que es un archivo
        if not file_path.is_file():
            result.add_error(f"La ruta no es un archivo: {file_path}", "type")
            return result
        
        # Validar tamaño
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / 1024 / 1024
        size_gb = size_mb / 1024
        
        if size_bytes < (self.min_file_size_kb * 1024):
            result.add_error(f"Archivo muy pequeño: {size_mb:.2f}MB", "size")
        
        if size_gb > self.max_file_size_gb:
            result.add_error(f"Archivo muy grande: {size_gb:.2f}GB (máximo {self.max_file_size_gb}GB)", "size")
        
        # Validar extensión según contexto
        expected_type = context.get('expected_type', 'any')
        if not self._validate_extension(file_path, expected_type):
            result.add_error(f"Extensión no válida para tipo '{expected_type}': {file_path.suffix}", "extension")
        
        # Validar permisos de lectura
        if not os.access(file_path, os.R_OK):
            result.add_error("Sin permisos de lectura", "permissions")
        
        # Agregar información del archivo al contexto
        result.context.update({
            'file_size_mb': round(size_mb, 2),
            'file_extension': file_path.suffix.lower(),
            'file_name': file_path.name,
            'file_stem': file_path.stem
        })
        
        return result
    
    def _validate_extension(self, file_path: Path, expected_type: str) -> bool:
        """Valida extensión según tipo esperado"""
        extension = file_path.suffix.lower()
        
        type_extensions = {
            'video': self.video_extensions,
            'audio': self.audio_extensions,
            'text': self.text_extensions,
            'any': self.video_extensions + self.audio_extensions + self.text_extensions
        }
        
        allowed_extensions = type_extensions.get(expected_type, [])
        return extension in allowed_extensions
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "existence": "El archivo debe existir",
            "size": f"Tamaño entre {self.min_file_size_kb}KB y {self.max_file_size_gb}GB",
            "permissions": "Debe tener permisos de lectura",
            "video_extensions": f"Extensiones de video: {', '.join(self.video_extensions)}",
            "audio_extensions": f"Extensiones de audio: {', '.join(self.audio_extensions)}",
            "text_extensions": f"Extensiones de texto: {', '.join(self.text_extensions)}"
        }

class ConfigValidator(IValidator):
    """Validador de configuración del sistema"""
    
    def __init__(self):
        self.required_keys = [
            'output_directory',
            'temp_directory'
        ]
        
        self.optional_keys = [
            'openai_api_key',
            'deepl_api_key',
            'max_concurrent_downloads',
            'default_audio_quality',
            'default_video_quality'
        ]
    
    def validate(self, config: Dict[str, Any], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """Valida configuración"""
        result = ValidationResult()
        
        if not isinstance(config, dict):
            result.add_error("Configuración debe ser un diccionario", "config")
            return result
        
        # Validar claves requeridas
        for key in self.required_keys:
            if key not in config:
                result.add_error(f"Clave requerida faltante: {key}", "required_keys")
            elif not config[key]:
                result.add_error(f"Valor requerido vacío: {key}", "required_values")
        
        # Validar directorios
        if 'output_directory' in config:
            dir_result = self._validate_directory(config['output_directory'], 'output_directory')
            result.merge(dir_result)
        
        if 'temp_directory' in config:
            dir_result = self._validate_directory(config['temp_directory'], 'temp_directory')
            result.merge(dir_result)
        
        # Validar API keys
        if 'openai_api_key' in config and config['openai_api_key']:
            if not self._validate_openai_key(config['openai_api_key']):
                result.add_warning("API key de OpenAI con formato inválido", "openai_api_key")
        
        if 'deepl_api_key' in config and config['deepl_api_key']:
            if not self._validate_deepl_key(config['deepl_api_key']):
                result.add_warning("API key de DeepL con formato inválido", "deepl_api_key")
        
        # Validar valores numéricos
        numeric_validations = {
            'max_concurrent_downloads': (1, 10),
            'default_audio_quality': (1, 5),
            'default_video_quality': (1, 5)
        }
        
        for key, (min_val, max_val) in numeric_validations.items():
            if key in config:
                if not self._validate_numeric_range(config[key], min_val, max_val):
                    result.add_error(f"{key} debe estar entre {min_val} y {max_val}", key)
        
        return result
    
    def _validate_directory(self, directory: str, field_name: str) -> ValidationResult:
        """Valida directorio"""
        result = ValidationResult()
        
        if not directory:
            result.add_error("Directorio no puede estar vacío", field_name)
            return result
        
        try:
            path = Path(directory)
            
            # Intentar crear directorio si no existe
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result.add_warning(f"Directorio creado: {directory}", field_name)
                except Exception as e:
                    result.add_error(f"No se pudo crear directorio: {e}", field_name)
                    return result
            
            # Validar permisos de escritura
            if not os.access(path, os.W_OK):
                result.add_error("Sin permisos de escritura en directorio", field_name)
            
        except Exception as e:
            result.add_error(f"Directorio inválido: {e}", field_name)
        
        return result
    
    def _validate_openai_key(self, api_key: str) -> bool:
        """Valida formato de API key de OpenAI"""
        return (api_key.startswith('sk-') and 
                len(api_key) > 45 and 
                api_key.replace('-', '').replace('_', '').isalnum())
    
    def _validate_deepl_key(self, api_key: str) -> bool:
        """Valida formato de API key de DeepL"""
        return (len(api_key) == 39 and 
                api_key.endswith(':fx') and
                api_key[:-3].replace('-', '').isalnum())
    
    def _validate_numeric_range(self, value: Any, min_val: int, max_val: int) -> bool:
        """Valida que un valor esté en un rango numérico"""
        try:
            num_value = int(value)
            return min_val <= num_value <= max_val
        except (ValueError, TypeError):
            return False
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "required_keys": f"Claves requeridas: {', '.join(self.required_keys)}",
            "directories": "Directorios deben ser válidos y escribibles",
            "api_keys": "API keys deben tener formato válido",
            "numeric_ranges": "Valores numéricos deben estar en rangos válidos"
        }

class ProjectValidator(IValidator):
    """Validador de estructura de proyectos"""
    
    def __init__(self):
        self.required_directories = [
            "1_original", "2_audio", "audio_separado",
            "3_transcripcion", "4_traduccion", "5_audio_es", "6_final"
        ]
        
        self.required_files = {
            "1_original": ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"],
            "2_audio": ["*.wav", "*.mp3"],
            "5_audio_es": ["*.wav", "*.mp3"]
        }
    
    def validate(self, project_path: Union[str, Path], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """Valida estructura de proyecto"""
        result = ValidationResult()
        context = context or {}
        
        if not project_path:
            result.add_error("Ruta de proyecto es requerida", "project_path")
            return result
        
        project_path = Path(project_path)
        
        # Validar existencia del proyecto
        if not project_path.exists():
            result.add_error(f"Directorio de proyecto no existe: {project_path}", "existence")
            return result
        
        if not project_path.is_dir():
            result.add_error(f"La ruta no es un directorio: {project_path}", "type")
            return result
        
        # Validar estructura de directorios
        for required_dir in self.required_directories:
            dir_path = project_path / required_dir
            if not dir_path.exists():
                result.add_error(f"Directorio faltante: {required_dir}", "structure")
            elif not dir_path.is_dir():
                result.add_error(f"No es directorio: {required_dir}", "structure")
        
        # Validar archivos requeridos
        for dir_name, file_patterns in self.required_files.items():
            dir_path = project_path / dir_name
            if dir_path.exists():
                found_files = []
                for pattern in file_patterns:
                    found_files.extend(list(dir_path.glob(pattern)))
                
                if not found_files:
                    expected_stage = context.get('expected_stage', 'unknown')
                    if self._should_have_files(dir_name, expected_stage):
                        result.add_error(f"No hay archivos en {dir_name}", "files")
                else:
                    result.context[f'{dir_name}_files'] = [f.name for f in found_files]
        
        # Validar metadata del proyecto
        metadata_file = project_path / "project_metadata.json"
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                result.context['metadata'] = metadata
            except Exception as e:
                result.add_warning(f"Error leyendo metadata: {e}", "metadata")
        else:
            result.add_warning("Archivo de metadata faltante", "metadata")
        
        # Calcular progreso del proyecto
        progress = self._calculate_project_progress(project_path)
        result.context['progress'] = progress
        
        return result
    
    def _should_have_files(self, directory: str, expected_stage: str) -> bool:
        """Determina si un directorio debería tener archivos según el stage esperado"""
        stage_requirements = {
            'downloaded': ['1_original'],
            'audio_extracted': ['1_original', '2_audio'],
            'audio_separated': ['1_original', '2_audio'],
            'transcribed': ['1_original', '2_audio'],
            'translated': ['1_original', '2_audio'],
            'tts_generated': ['1_original', '2_audio', '5_audio_es'],
            'video_composed': ['1_original', '2_audio', '5_audio_es']
        }
        
        required_dirs = stage_requirements.get(expected_stage, [])
        return directory in required_dirs
    
    def _calculate_project_progress(self, project_path: Path) -> Dict[str, Any]:
        """Calcula progreso del proyecto"""
        steps = {
            'video_downloaded': self._has_files(project_path / "1_original", ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]),
            'audio_extracted': self._has_files(project_path / "2_audio", ["*.wav", "*.mp3"]),
            'audio_separated': (
                self._has_files(project_path / "audio_separado", ["*vocals*.wav"]) and
                self._has_files(project_path / "audio_separado", ["*accompaniment*.wav"])
            ),
            'transcribed': self._has_files(project_path / "3_transcripcion", ["*.txt", "*.srt"]),
            'translated': self._has_files(project_path / "4_traduccion", ["*.txt"]),
            'spanish_audio': self._has_files(project_path / "5_audio_es", ["*.wav", "*.mp3"]),
            'final_video': self._has_files(project_path / "6_final", ["*.mp4", "*.avi", "*.mkv"])
        }
        
        completed = sum(steps.values())
        total = len(steps)
        percentage = (completed / total) * 100 if total > 0 else 0
        
        return {
            'steps': steps,
            'completed_steps': completed,
            'total_steps': total,
            'percentage': round(percentage, 1),
            'is_complete': completed == total
        }
    
    def _has_files(self, directory: Path, patterns: List[str]) -> bool:
        """Verifica si un directorio tiene archivos que coincidan con los patrones"""
        if not directory.exists():
            return False
        
        for pattern in patterns:
            if list(directory.glob(pattern)):
                return True
        
        return False
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "structure": f"Directorios requeridos: {', '.join(self.required_directories)}",
            "files": "Archivos apropiados en cada directorio según el progreso",
            "metadata": "Archivo de metadata del proyecto (recomendado)",
            "progress": "Validación basada en el progreso esperado del proyecto"
        }

class VideoProcessingRequestValidator(IValidator):
    """Validador específico para solicitudes de procesamiento de video"""
    
    def __init__(self):
        self.url_validator = URLValidator()
        self.file_validator = FileValidator()
        self.config_validator = ConfigValidator()
    
    def validate(self, request: Dict[str, Any], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """Valida solicitud completa de procesamiento"""
        result = ValidationResult()
        
        # Validar estructura básica de la solicitud
        required_fields = ['video_url', 'output_directory']
        optional_fields = ['voice_preference', 'transcription_model', 'force_language']
        
        for field in required_fields:
            if field not in request:
                result.add_error(f"Campo requerido faltante: {field}", field)
            elif not request[field]:
                result.add_error(f"Campo requerido vacío: {field}", field)
        
        # Validar URL del video
        if 'video_url' in request and request['video_url']:
            url_result = self.url_validator.validate(request['video_url'])
            if not url_result.is_valid:
                for error in url_result.errors:
                    result.add_error(error, 'video_url')
            result.context.update(url_result.context)
        
        # Validar directorio de salida
        if 'output_directory' in request and request['output_directory']:
            dir_result = self._validate_output_directory(request['output_directory'])
            result.merge(dir_result)
        
        # Validar preferencias opcionales
        if 'voice_preference' in request and request['voice_preference']:
            if not self._validate_voice_preference(request['voice_preference']):
                result.add_warning("Preferencia de voz no reconocida", 'voice_preference')
        
        if 'transcription_model' in request and request['transcription_model']:
            if not self._validate_transcription_model(request['transcription_model']):
                result.add_error("Modelo de transcripción no válido", 'transcription_model')
        
        if 'force_language' in request and request['force_language']:
            if not self._validate_language_code(request['force_language']):
                result.add_error("Código de idioma no válido", 'force_language')
        
        return result
    
    def _validate_output_directory(self, output_dir: str) -> ValidationResult:
        """Valida directorio de salida"""
        result = ValidationResult()
        
        try:
            path = Path(output_dir)
            
            # Crear directorio si no existe
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                result.add_warning(f"Directorio creado: {output_dir}", 'output_directory')
            
            # Verificar permisos de escritura
            if not os.access(path, os.W_OK):
                result.add_error("Sin permisos de escritura en directorio de salida", 'output_directory')
            
            # Verificar espacio disponible (básico)
            stat = os.statvfs(path)
            free_space_gb = (stat.f_frsize * stat.f_avail) / (1024**3)
            
            if free_space_gb < 1.0:  # Menos de 1GB
                result.add_warning(f"Poco espacio disponible: {free_space_gb:.1f}GB", 'output_directory')
            
            result.context['free_space_gb'] = round(free_space_gb, 2)
            
        except Exception as e:
            result.add_error(f"Error validando directorio: {e}", 'output_directory')
        
        return result
    
    def _validate_voice_preference(self, voice_pref: str) -> bool:
        """Valida preferencia de voz"""
        valid_preferences = ['Male', 'Female', 'male', 'female']
        # También aceptar IDs específicos de voces (formato: idioma-región-nombre)
        voice_id_pattern = r'^[a-z]{2}-[A-Z]{2}-[A-Za-z]+Neural$'
        
        return (voice_pref in valid_preferences or 
                re.match(voice_id_pattern, voice_pref))
    
    def _validate_transcription_model(self, model: str) -> bool:
        """Valida modelo de transcripción"""
        valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3']
        return model in valid_models
    
    def _validate_language_code(self, lang_code: str) -> bool:
        """Valida código de idioma"""
        # Lista básica de códigos de idioma ISO 639-1
        valid_codes = [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
            'th', 'vi', 'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'he', 'cs', 'sk',
            'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'uk', 'be', 'mk'
        ]
        return lang_code.lower() in valid_codes
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "video_url": "URL válida de video de plataforma soportada",
            "output_directory": "Directorio válido con permisos de escritura",
            "voice_preference": "Male, Female, o ID específico de voz",
            "transcription_model": "Modelo Whisper válido (tiny, base, small, medium, large, etc.)",
            "force_language": "Código de idioma ISO 639-1 válido",
            "space_requirements": "Al menos 1GB de espacio libre recomendado"
        }

class VideoCompositionRequestValidator(IValidator):
    """Validador para solicitudes de composición de video"""
    
    def __init__(self):
        self.file_validator = FileValidator()
    
    def validate(self, request: Dict[str, Any], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """Valida solicitud de composición de video"""
        result = ValidationResult()
        
        # Campos requeridos
        required_files = ['original_video_path', 'spanish_audio_path', 'output_path']
        
        for field in required_files:
            if field not in request:
                result.add_error(f"Campo requerido: {field}", field)
                continue
            
            if not request[field]:
                result.add_error(f"Ruta vacía: {field}", field)
                continue
            
            # Validar archivos específicos
            if field == 'original_video_path':
                file_result = self.file_validator.validate(
                    request[field], 
                    {'expected_type': 'video'}
                )
            elif field == 'spanish_audio_path':
                file_result = self.file_validator.validate(
                    request[field], 
                    {'expected_type': 'audio'}
                )
            else:  # output_path
                # Para output_path, validar que el directorio padre exista
                output_path = Path(request[field])
                if not output_path.parent.exists():
                    result.add_error(f"Directorio padre no existe: {output_path.parent}", field)
                continue
            
            if not file_result.is_valid:
                for error in file_result.errors:
                    result.add_error(error, field)
        
        # Validar archivo opcional de música de fondo
        if 'background_music_path' in request and request['background_music_path']:
            music_result = self.file_validator.validate(
                request['background_music_path'],
                {'expected_type': 'audio'}
            )
            if not music_result.is_valid:
                for error in music_result.errors:
                    result.add_warning(f"Música de fondo: {error}", 'background_music_path')
        
        # Validar parámetros de volumen
        volume_params = ['spanish_voice_volume', 'background_music_volume']
        for param in volume_params:
            if param in request:
                try:
                    volume = float(request[param])
                    if not 0.0 <= volume <= 1.0:
                        result.add_error(f"{param} debe estar entre 0.0 y 1.0", param)
                except (ValueError, TypeError):
                    result.add_error(f"{param} debe ser un número", param)
        
        return result
    
    def get_validation_rules(self) -> Dict[str, str]:
        """Retorna reglas de validación"""
        return {
            "original_video_path": "Archivo de video válido y existente",
            "spanish_audio_path": "Archivo de audio válido y existente",
            "output_path": "Ruta de salida válida (directorio padre debe existir)",
            "background_music_path": "Archivo de audio opcional para música de fondo",
            "volume_levels": "Valores de volumen entre 0.0 y 1.0"
        }

class CompositeValidator:
    """Validador compuesto que combina múltiples validadores"""
    
    def __init__(self):
        self.validators: Dict[str, IValidator] = {}
    
    def add_validator(self, name: str, validator: IValidator):
        """Agrega un validador"""
        self.validators[name] = validator
    
    def validate_all(self, data: Dict[str, Any], 
                    context: Dict[str, Any] = None) -> ValidationResult:
        """Valida usando todos los validadores"""
        combined_result = ValidationResult()
        
        for name, validator in self.validators.items():
            if name in data:
                result = validator.validate(data[name], context)
                result.errors = [f"[{name}] {error}" for error in result.errors]
                result.warnings = [f"[{name}] {warning}" for warning in result.warnings]
                combined_result.merge(result)
        
        return combined_result
    
    def validate_specific(self, validator_name: str, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Valida usando un validador específico"""
        if validator_name not in self.validators:
            result = ValidationResult()
            result.add_error(f"Validador no encontrado: {validator_name}")
            return result
        
        return self.validators[validator_name].validate(value, context)
    
    def get_all_validation_rules(self) -> Dict[str, Dict[str, str]]:
        """Obtiene todas las reglas de validación"""
        all_rules = {}
        for name, validator in self.validators.items():
            all_rules[name] = validator.get_validation_rules()
        return all_rules

# Funciones de conveniencia para validaciones rápidas

def validate_video_url(url: str) -> Tuple[bool, List[str]]:
    """Validación rápida de URL de video"""
    validator = URLValidator()
    result = validator.validate(url)
    return result.is_valid, result.errors

def validate_file_exists(file_path: Union[str, Path], 
                        expected_type: str = 'any') -> Tuple[bool, List[str]]:
    """Validación rápida de existencia de archivo"""
    validator = FileValidator()
    result = validator.validate(file_path, {'expected_type': expected_type})
    return result.is_valid, result.errors

def validate_project_structure(project_path: Union[str, Path], expected_stage: str = 'unknown') -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validación rápida de estructura de proyecto"""
    validator = ProjectValidator()
    result = validator.validate(project_path, {'expected_stage': expected_stage})
    return result.is_valid, result.errors, result.context

def validate_config_dict(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validación rápida de configuración"""
    validator = ConfigValidator()
    result = validator.validate(config)
    return result.is_valid, result.errors, result.warnings

def validate_operation_context(operation: str, context: Dict[str, Any]) -> ValidationResult:
    """Valida contexto para operaciones específicas"""
    result = ValidationResult()
    
    operation_requirements = {
        'download': ['video_url', 'output_directory'],
        'audio_processing': ['video_file', 'output_directory'],
        'transcription': ['audio_file', 'output_directory'],
        'translation': ['text', 'source_language', 'target_language'],
        'tts': ['text', 'output_file', 'voice_preference'],
        'composition': ['original_video', 'spanish_audio', 'output_file']
    }
    
    required_fields = operation_requirements.get(operation, [])
    
    for field in required_fields:
        if field not in context:
            result.add_error(f"Campo requerido para {operation}: {field}", field)
        elif not context[field]:
            result.add_error(f"Valor vacío para {operation}: {field}", field)
    
    return result

# Factory para crear validadores
def create_validator_suite() -> CompositeValidator:
    """Crea un conjunto completo de validadores"""
    suite = CompositeValidator()
    
    suite.add_validator('url', URLValidator())
    suite.add_validator('file', FileValidator())
    suite.add_validator('config', ConfigValidator())
    suite.add_validator('project', ProjectValidator())
    suite.add_validator('video_request', VideoProcessingRequestValidator())
    suite.add_validator('composition_request', VideoCompositionRequestValidator())
    
    return suite

# Función de validación rápida para traducción
def quick_validate_for_translation(input_source: str, output_dir: str) -> Tuple[bool, List[str]]:
    """
    Validación rápida para inicio de traducción
    
    Args:
        input_source: URL de video o ruta de archivo
        output_dir: Directorio de salida
        
    Returns:
        Tuple con (es_válido, lista_de_errores)
    """
    errors = []
    
    # Detectar si es URL o archivo
    if input_source.startswith(('http://', 'https://')):
        # Validar URL
        is_valid, url_errors = validate_video_url(input_source)
        if not is_valid:
            errors.extend(url_errors)
    else:
        # Validar archivo local
        is_valid, file_errors = validate_file_exists(input_source, 'video')
        if not is_valid:
            errors.extend(file_errors)
    
    # Validar directorio de salida
    try:
        output_path = Path(output_dir)
        if not output_path.parent.exists():
            errors.append(f"Directorio padre no existe: {output_path.parent}")
        
        # Intentar crear si no existe
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar permisos
        if not os.access(output_path, os.W_OK):
            errors.append(f"Sin permisos de escritura en: {output_dir}")
            
    except Exception as e:
        errors.append(f"Error con directorio de salida: {e}")
    
    return len(errors) == 0, errors