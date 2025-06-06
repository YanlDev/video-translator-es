"""
MÓDULO: GESTIÓN DE CONFIGURACIÓN
===============================
Centraliza toda la lógica de configuración del sistema
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime

from .error_handler import ConfigurationError, ErrorSeverity
from .validators import ConfigValidator, ValidationResult

@dataclass
class EnvironmentConfig:
    """Configuración desde variables de entorno"""
    openai_api_key: Optional[str] = None
    deepl_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    
    # Configuración de directorios
    output_directory: str = "downloads"
    temp_directory: str = "temp"
    cache_directory: str = "cache"
    
    # Configuración de procesamiento
    max_concurrent_downloads: int = 3
    default_transcription_model: str = "base"
    default_voice_gender: str = "Female"
    default_target_language: str = "es"
    
    # Configuración de calidad
    default_audio_quality: int = 3  # 1-5
    default_video_quality: int = 3  # 1-5
    max_file_size_gb: float = 5.0
    
    # Configuración de red
    request_timeout: int = 30
    max_retries: int = 3
    enable_proxy: bool = False
    proxy_url: Optional[str] = None
    
    # Configuración de logging
    log_level: str = "INFO"
    log_file: str = "video_processing.log"
    enable_file_logging: bool = True
    
    @classmethod
    def from_environment(cls) -> 'EnvironmentConfig':
        """Crea configuración desde variables de entorno"""
        return cls(
            # API Keys
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            deepl_api_key=os.getenv('DEEPL_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            azure_api_key=os.getenv('AZURE_API_KEY'),
            
            # Directorios
            output_directory=os.getenv('VIDEO_OUTPUT_DIR', 'downloads'),
            temp_directory=os.getenv('VIDEO_TEMP_DIR', 'temp'),
            cache_directory=os.getenv('VIDEO_CACHE_DIR', 'cache'),
            
            # Procesamiento
            max_concurrent_downloads=int(os.getenv('MAX_CONCURRENT_DOWNLOADS', '3')),
            default_transcription_model=os.getenv('DEFAULT_TRANSCRIPTION_MODEL', 'base'),
            default_voice_gender=os.getenv('DEFAULT_VOICE_GENDER', 'Female'),
            default_target_language=os.getenv('DEFAULT_TARGET_LANGUAGE', 'es'),
            
            # Calidad
            default_audio_quality=int(os.getenv('DEFAULT_AUDIO_QUALITY', '3')),
            default_video_quality=int(os.getenv('DEFAULT_VIDEO_QUALITY', '3')),
            max_file_size_gb=float(os.getenv('MAX_FILE_SIZE_GB', '5.0')),
            
            # Red
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            enable_proxy=os.getenv('ENABLE_PROXY', 'false').lower() == 'true',
            proxy_url=os.getenv('PROXY_URL'),
            
            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'video_processing.log'),
            enable_file_logging=os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
        )

@dataclass
class SystemConfig:
    """Configuración del sistema"""
    
    # Información del sistema
    app_name: str = "Video Processing System"
    app_version: str = "1.0.0"
    config_version: str = "1.0"
    
    # Configuración de componentes
    enable_gpu_acceleration: bool = True
    enable_audio_separation: bool = True
    enable_background_music: bool = True
    enable_subtitles: bool = True
    
    # Configuración de proveedores
    preferred_translation_provider: str = "adaptive"  # openai, google, deepl, adaptive
    preferred_tts_provider: str = "edge"  # edge, google, azure
    preferred_transcription_provider: str = "whisper"  # whisper, google
    
    # Configuración de calidad vs velocidad
    quality_preset: str = "balanced"  # fast, balanced, quality
    
    # Configuración de archivos temporales
    cleanup_temp_files: bool = True
    keep_intermediate_files: bool = False
    
    # Configuración de notificaciones
    enable_notifications: bool = True
    enable_progress_display: bool = True
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Obtiene configuración basada en preset de calidad"""
        quality_presets = {
            "fast": {
                "transcription_model": "tiny",
                "audio_quality": 2,
                "video_quality": 2,
                "enable_audio_separation": False,
                "translation_provider": "google"
            },
            "balanced": {
                "transcription_model": "base",
                "audio_quality": 3,
                "video_quality": 3,
                "enable_audio_separation": True,
                "translation_provider": "adaptive"
            },
            "quality": {
                "transcription_model": "large-v3",
                "audio_quality": 5,
                "video_quality": 5,
                "enable_audio_separation": True,
                "translation_provider": "openai"
            }
        }
        
        return quality_presets.get(self.quality_preset, quality_presets["balanced"])

class IConfigLoader(ABC):
    """Interface para cargadores de configuración"""
    
    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """Carga configuración desde fuente"""
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """Guarda configuración a destino"""
        pass
    
    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """Verifica si soporta el formato del archivo"""
        pass

class JSONConfigLoader(IConfigLoader):
    """Cargador de configuración JSON"""
    
    def load(self, source: str) -> Dict[str, Any]:
        """Carga configuración desde archivo JSON"""
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Archivo de configuración no encontrado: {source}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Error parseando JSON: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error cargando configuración: {e}")
    
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """Guarda configuración a archivo JSON"""
        try:
            # Crear directorio si no existe
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            raise ConfigurationError(f"Error guardando configuración JSON: {e}")
    
    def supports_format(self, file_path: str) -> bool:
        """Verifica si es archivo JSON"""
        return file_path.lower().endswith('.json')

class YAMLConfigLoader(IConfigLoader):
    """Cargador de configuración YAML"""
    
    def load(self, source: str) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML"""
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigurationError(f"Archivo de configuración no encontrado: {source}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parseando YAML: {e}")
        except ImportError:
            raise ConfigurationError("PyYAML no está instalado. Usar: pip install PyYAML")
        except Exception as e:
            raise ConfigurationError(f"Error cargando configuración YAML: {e}")
    
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """Guarda configuración a archivo YAML"""
        try:
            # Crear directorio si no existe
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, 
                              allow_unicode=True, sort_keys=False)
            return True
        except ImportError:
            raise ConfigurationError("PyYAML no está instalado. Usar: pip install PyYAML")
        except Exception as e:
            raise ConfigurationError(f"Error guardando configuración YAML: {e}")
    
    def supports_format(self, file_path: str) -> bool:
        """Verifica si es archivo YAML"""
        return file_path.lower().endswith(('.yaml', '.yml'))

class ConfigManager:
    """Gestor principal de configuración del sistema"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuraciones cargadas
        self._environment_config: Optional[EnvironmentConfig] = None
        self._system_config: Optional[SystemConfig] = None
        self._user_config: Dict[str, Any] = {}
        
        # Cargadores disponibles
        self.loaders: Dict[str, IConfigLoader] = {
            'json': JSONConfigLoader(),
            'yaml': YAMLConfigLoader()
        }
        
        # Validador
        self.validator = ConfigValidator()
        
        # Cargar configuraciones
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Carga todas las configuraciones disponibles"""
        # 1. Configuración desde variables de entorno
        self._environment_config = EnvironmentConfig.from_environment()
        
        # 2. Configuración del sistema (defaults + archivo)
        self._system_config = self._load_system_config()
        
        # 3. Configuración de usuario (personalizada)
        self._user_config = self._load_user_config()
    
    def _load_system_config(self) -> SystemConfig:
        """Carga configuración del sistema"""
        system_config = SystemConfig()
        
        # Intentar cargar desde archivo
        config_files = [
            self.config_dir / "system.json",
            self.config_dir / "system.yaml",
            self.config_dir / "system.yml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    loader = self._get_loader_for_file(str(config_file))
                    if loader:
                        file_config = loader.load(str(config_file))
                        # Actualizar configuración con valores del archivo
                        for key, value in file_config.items():
                            if hasattr(system_config, key):
                                setattr(system_config, key, value)
                        break
                except Exception as e:
                    print(f"⚠️  Error cargando {config_file}: {e}")
        
        return system_config
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Carga configuración de usuario"""
        user_config = {}
        
        # Intentar cargar desde archivo
        config_files = [
            self.config_dir / "user.json",
            self.config_dir / "user.yaml",
            self.config_dir / "user.yml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    loader = self._get_loader_for_file(str(config_file))
                    if loader:
                        user_config = loader.load(str(config_file))
                        break
                except Exception as e:
                    print(f"⚠️  Error cargando {config_file}: {e}")
        
        return user_config
    
    def _get_loader_for_file(self, file_path: str) -> Optional[IConfigLoader]:
        """Obtiene cargador apropiado para un archivo"""
        for loader in self.loaders.values():
            if loader.supports_format(file_path):
                return loader
        return None
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Obtiene valor de configuración con precedencia"""
        # Precedencia: user_config > environment_config > system_config > default
        
        # 1. Configuración de usuario
        if key in self._user_config:
            return self._user_config[key]
        
        # 2. Configuración de entorno
        if hasattr(self._environment_config, key):
            value = getattr(self._environment_config, key)
            if value is not None:
                return value
        
        # 3. Configuración del sistema
        if hasattr(self._system_config, key):
            value = getattr(self._system_config, key)
            if value is not None:
                return value
        
        # 4. Valor por defecto
        return default
    
    def set_config(self, key: str, value: Any, persist: bool = True) -> bool:
        """Establece valor de configuración"""
        try:
            # Actualizar configuración en memoria
            self._user_config[key] = value
            
            # Persistir si se solicita
            if persist:
                return self.save_user_config()
            
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Error estableciendo configuración {key}: {e}")
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Obtiene configuración completa mezclada"""
        merged = {}
        
        # Empezar con configuración del sistema
        if self._system_config:
            merged.update(asdict(self._system_config))
        
        # Agregar configuración de entorno
        if self._environment_config:
            env_dict = asdict(self._environment_config)
            for key, value in env_dict.items():
                if value is not None:
                    merged[key] = value
        
        # Agregar configuración de usuario
        merged.update(self._user_config)
        
        return merged
    
    def validate_config(self) -> ValidationResult:
        """Valida configuración actual"""
        merged_config = self.get_merged_config()
        return self.validator.validate(merged_config)
    
    def save_user_config(self, format: str = 'json') -> bool:
        """Guarda configuración de usuario"""
        try:
            if format not in self.loaders:
                raise ConfigurationError(f"Formato no soportado: {format}")
            
            filename = f"user.{format}"
            file_path = self.config_dir / filename
            
            # Agregar metadata
            config_to_save = {
                '_metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'config_version': self._system_config.config_version,
                    'app_version': self._system_config.app_version
                },
                **self._user_config
            }
            
            loader = self.loaders[format]
            return loader.save(config_to_save, str(file_path))
            
        except Exception as e:
            raise ConfigurationError(f"Error guardando configuración de usuario: {e}")
    
    def create_default_config_files(self):
        """Crea archivos de configuración por defecto"""
        try:
            # Crear configuración del sistema por defecto
            system_config_path = self.config_dir / "system.json"
            if not system_config_path.exists():
                system_dict = asdict(SystemConfig())
                self.loaders['json'].save(system_dict, str(system_config_path))
                print(f"✅ Creado: {system_config_path}")
            
            # Crear configuración de usuario por defecto
            user_config_path = self.config_dir / "user.json"
            if not user_config_path.exists():
                default_user_config = {
                    '_metadata': {
                        'created_at': datetime.now().isoformat(),
                        'description': 'Configuración personalizada del usuario'
                    },
                    'preferred_voice_gender': 'Female',
                    'preferred_transcription_model': 'base',
                    'enable_notifications': True,
                    'custom_output_directory': None
                }
                self.loaders['json'].save(default_user_config, str(user_config_path))
                print(f"✅ Creado: {user_config_path}")
            
            # Crear archivo de ejemplo de configuración de entorno
            env_example_path = self.config_dir / ".env.example"
            if not env_example_path.exists():
                env_example_content = """# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# Copiar a .env y completar con valores reales

# API Keys (opcionales)
OPENAI_API_KEY=sk-your-openai-key-here
DEEPL_API_KEY=your-deepl-key-here
GOOGLE_API_KEY=your-google-key-here

# Directorios
VIDEO_OUTPUT_DIR=downloads
VIDEO_TEMP_DIR=temp
VIDEO_CACHE_DIR=cache

# Configuración de procesamiento
MAX_CONCURRENT_DOWNLOADS=3
DEFAULT_TRANSCRIPTION_MODEL=base
DEFAULT_VOICE_GENDER=Female
DEFAULT_TARGET_LANGUAGE=es

# Configuración de calidad (1-5)
DEFAULT_AUDIO_QUALITY=3
DEFAULT_VIDEO_QUALITY=3
MAX_FILE_SIZE_GB=5.0

# Configuración de red
REQUEST_TIMEOUT=30
MAX_RETRIES=3
ENABLE_PROXY=false
# PROXY_URL=http://proxy.example.com:8080

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=video_processing.log
ENABLE_FILE_LOGGING=true
"""
                
                with open(env_example_path, 'w', encoding='utf-8') as f:
                    f.write(env_example_content)
                print(f"✅ Creado: {env_example_path}")
                
        except Exception as e:
            raise ConfigurationError(f"Error creando archivos de configuración: {e}")
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Obtiene todas las API keys configuradas"""
        return {
            'openai': self.get_config('openai_api_key'),
            'deepl': self.get_config('deepl_api_key'),
            'google': self.get_config('google_api_key'),
            'azure': self.get_config('azure_api_key')
        }
    
    def get_directory_config(self) -> Dict[str, str]:
        """Obtiene configuración de directorios"""
        return {
            'output': self.get_config('output_directory', 'downloads'),
            'temp': self.get_config('temp_directory', 'temp'),
            'cache': self.get_config('cache_directory', 'cache')
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento"""
        return {
            'max_concurrent_downloads': self.get_config('max_concurrent_downloads', 3),
            'default_transcription_model': self.get_config('default_transcription_model', 'base'),
            'default_voice_gender': self.get_config('default_voice_gender', 'Female'),
            'default_target_language': self.get_config('default_target_language', 'es'),
            'default_audio_quality': self.get_config('default_audio_quality', 3),
            'default_video_quality': self.get_config('default_video_quality', 3),
            'max_file_size_gb': self.get_config('max_file_size_gb', 5.0)
        }
    
    def get_network_config(self) -> Dict[str, Any]:
        """Obtiene configuración de red"""
        return {
            'request_timeout': self.get_config('request_timeout', 30),
            'max_retries': self.get_config('max_retries', 3),
            'enable_proxy': self.get_config('enable_proxy', False),
            'proxy_url': self.get_config('proxy_url')
        }
    
    def is_api_configured(self, service: str) -> bool:
        """Verifica si un servicio tiene API key configurada"""
        key = self.get_config(f'{service}_api_key')
        return key is not None and len(key.strip()) > 0
    
    def get_quality_preset_config(self) -> Dict[str, Any]:
        """Obtiene configuración basada en preset de calidad"""
        quality_preset = self.get_config('quality_preset', 'balanced')
        
        if self._system_config:
            # Temporalmente cambiar el preset para obtener la configuración
            original_preset = self._system_config.quality_preset
            self._system_config.quality_preset = quality_preset
            config = self._system_config.get_quality_settings()
            self._system_config.quality_preset = original_preset
            return config
        
        # Fallback si no hay configuración del sistema
        return {
            "transcription_model": "base",
            "audio_quality": 3,
            "video_quality": 3,
            "enable_audio_separation": True,
            "translation_provider": "adaptive"
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any], validate: bool = True) -> ValidationResult:
        """Actualiza configuración desde diccionario"""
        if validate:
            validation_result = self.validator.validate(config_dict)
            if not validation_result.is_valid:
                return validation_result
        
        # Actualizar configuración de usuario
        for key, value in config_dict.items():
            if not key.startswith('_'):  # Ignorar metadata
                self._user_config[key] = value
        
        return ValidationResult(True)
    
    def reset_to_defaults(self) -> bool:
        """Resetea configuración de usuario a valores por defecto"""
        try:
            self._user_config.clear()
            return self.save_user_config()
        except Exception as e:
            raise ConfigurationError(f"Error reseteando configuración: {e}")
    
    def export_config(self, file_path: str, include_sensitive: bool = False) -> bool:
        """Exporta configuración completa a archivo"""
        try:
            loader = self._get_loader_for_file(file_path)
            if not loader:
                raise ConfigurationError(f"Formato de archivo no soportado: {file_path}")
            
            config = self.get_merged_config()
            
            # Filtrar información sensible si se solicita
            if not include_sensitive:
                sensitive_keys = ['openai_api_key', 'deepl_api_key', 'google_api_key', 'azure_api_key']
                for key in sensitive_keys:
                    if key in config and config[key]:
                        config[key] = "***HIDDEN***"
            
            # Agregar metadata de exportación
            config['_export_metadata'] = {
                'exported_at': datetime.now().isoformat(),
                'exported_by': 'ConfigManager',
                'includes_sensitive': include_sensitive
            }
            
            return loader.save(config, file_path)
            
        except Exception as e:
            raise ConfigurationError(f"Error exportando configuración: {e}")
    
    def import_config(self, file_path: str, merge: bool = True) -> ValidationResult:
        """Importa configuración desde archivo"""
        try:
            if not Path(file_path).exists():
                raise ConfigurationError(f"Archivo no encontrado: {file_path}")
            
            loader = self._get_loader_for_file(file_path)
            if not loader:
                raise ConfigurationError(f"Formato de archivo no soportado: {file_path}")
            
            imported_config = loader.load(file_path)
            
            # Remover metadata de exportación
            imported_config.pop('_export_metadata', None)
            imported_config.pop('_metadata', None)
            
            # Validar configuración importada
            validation_result = self.validator.validate(imported_config)
            if not validation_result.is_valid:
                return validation_result
            
            # Aplicar configuración
            if merge:
                # Mergear con configuración existente
                self._user_config.update(imported_config)
            else:
                # Reemplazar configuración completa
                self._user_config = imported_config.copy()
            
            return ValidationResult(True)
            
        except Exception as e:
            result = ValidationResult(False)
            result.add_error(f"Error importando configuración: {e}")
            return result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de configuración actual"""
        api_keys = self.get_api_keys()
        
        return {
            'config_status': {
                'environment_loaded': self._environment_config is not None,
                'system_loaded': self._system_config is not None,
                'user_config_items': len(self._user_config),
                'validation_status': self.validate_config().is_valid
            },
            'api_keys_configured': {
                service: key is not None and len(key.strip()) > 0 
                for service, key in api_keys.items()
            },
            'directories': self.get_directory_config(),
            'quality_preset': self.get_config('quality_preset', 'balanced'),
            'enabled_features': {
                'gpu_acceleration': self.get_config('enable_gpu_acceleration', True),
                'audio_separation': self.get_config('enable_audio_separation', True),
                'background_music': self.get_config('enable_background_music', True),
                'subtitles': self.get_config('enable_subtitles', True)
            }
        }

# Funciones de conveniencia para configuración

def get_default_config_manager() -> ConfigManager:
    """Crea gestor de configuración con valores por defecto"""
    return ConfigManager()

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo"""
    config_manager = ConfigManager()
    
    if file_path.endswith('.json'):
        loader = JSONConfigLoader()
    elif file_path.endswith(('.yaml', '.yml')):
        loader = YAMLConfigLoader()
    else:
        raise ConfigurationError(f"Formato de archivo no soportado: {file_path}")
    
    return loader.load(file_path)

def create_config_from_template() -> ConfigManager:
    """Crea configuración desde template y la guarda"""
    config_manager = ConfigManager()
    config_manager.create_default_config_files()
    return config_manager

def validate_environment_setup() -> ValidationResult:
    """Valida que el entorno esté configurado correctamente"""
    result = ValidationResult()
    
    try:
        # Verificar configuración básica
        config_manager = ConfigManager()
        validation = config_manager.validate_config()
        
        if not validation.is_valid:
            result.merge(validation)
            return result
        
        # Verificar directorios
        directories = config_manager.get_directory_config()
        for dir_type, dir_path in directories.items():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                if not os.access(dir_path, os.W_OK):
                    result.add_error(f"Sin permisos de escritura en {dir_type}: {dir_path}")
            except Exception as e:
                result.add_error(f"Error con directorio {dir_type}: {e}")
        
        # Verificar dependencias opcionales
        api_keys = config_manager.get_api_keys()
        if not any(api_keys.values()):
            result.add_warning("No hay API keys configuradas - funcionalidad limitada")
        
        if not result.errors:
            result.add_warning("Entorno configurado correctamente") if result.warnings else None
        
    except Exception as e:
        result.add_error(f"Error validando entorno: {e}")
    
    return result

# Context manager para configuración temporal
class TemporaryConfig:
    """Context manager para cambios temporales de configuración"""
    
    def __init__(self, config_manager: ConfigManager, temp_config: Dict[str, Any]):
        self.config_manager = config_manager
        self.temp_config = temp_config
        self.original_config = {}
    
    def __enter__(self):
        # Guardar configuración original
        for key in self.temp_config:
            self.original_config[key] = self.config_manager.get_config(key)
        
        # Aplicar configuración temporal
        for key, value in self.temp_config.items():
            self.config_manager.set_config(key, value, persist=False)
        
        return self.config_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar configuración original
        for key, value in self.original_config.items():
            if value is not None:
                self.config_manager.set_config(key, value, persist=False)
            else:
                # Remover clave si no existía originalmente
                self.config_manager._user_config.pop(key, None)