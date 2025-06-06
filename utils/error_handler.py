"""
MÓDULO: MANEJO DE ERRORES DEL SISTEMA
=====================================
Centraliza toda la lógica de manejo de errores y excepciones
"""

import traceback
import logging
import sys
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ErrorSeverity(Enum):
    """Niveles de severidad de errores"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categorías de errores del sistema"""
    DOWNLOAD = "download"
    AUDIO_PROCESSING = "audio_processing"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    TTS = "tts"
    VIDEO_COMPOSITION = "video_composition"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"

@dataclass
class ErrorInfo:
    """Información detallada de un error"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: str
    timestamp: datetime
    user_message: str
    suggested_actions: List[str]
    context: Dict[str, Any] = None
    stacktrace: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class VideoProcessingError(Exception):
    """Excepción base para errores de procesamiento de video"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()

class ValidationError(VideoProcessingError):
    """Error de validación de datos"""
    
    def __init__(self, message: str, field: str = "", value: Any = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM)
        self.field = field
        self.value = value

class NetworkError(VideoProcessingError):
    """Error de conexión de red"""
    
    def __init__(self, message: str, url: str = "", status_code: int = 0):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.HIGH)
        self.url = url
        self.status_code = status_code

class FileSystemError(VideoProcessingError):
    """Error del sistema de archivos"""
    
    def __init__(self, message: str, file_path: str = "", operation: str = ""):
        super().__init__(message, ErrorCategory.FILE_SYSTEM, ErrorSeverity.HIGH)
        self.file_path = file_path
        self.operation = operation

class ConfigurationError(VideoProcessingError):
    """Error de configuración"""
    
    def __init__(self, message: str, config_key: str = ""):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH)
        self.config_key = config_key

class IErrorLogger(ABC):
    """Interface para loggers de errores"""
    
    @abstractmethod
    def log_error(self, error_info: ErrorInfo) -> None:
        """Registra un error"""
        pass
    
    @abstractmethod
    def get_error_history(self, limit: int = 100) -> List[ErrorInfo]:
        """Obtiene historial de errores"""
        pass

class IErrorNotifier(ABC):
    """Interface para notificadores de errores"""
    
    @abstractmethod
    def notify_error(self, error_info: ErrorInfo) -> bool:
        """Notifica un error"""
        pass

class ConsoleErrorLogger(IErrorLogger):
    """Logger de errores a consola"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura logging básico"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_error(self, error_info: ErrorInfo) -> None:
        """Registra error en consola y historial"""
        self.error_history.append(error_info)
        
        # Determinar nivel de logging
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        # Crear mensaje de log
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        if error_info.details:
            log_message += f" - {error_info.details}"
        
        self.logger.log(log_level, log_message)
        
        # Mostrar en consola con formato visual
        self._display_error_console(error_info)
    
    def _display_error_console(self, error_info: ErrorInfo):
        """Muestra error formateado en consola"""
        severity_icons = {
            ErrorSeverity.LOW: "ℹ️",
            ErrorSeverity.MEDIUM: "⚠️",
            ErrorSeverity.HIGH: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(error_info.severity, "❌")
        
        print(f"\n{icon} {error_info.severity.value.upper()} - {error_info.category.value}")
        print(f"   Mensaje: {error_info.user_message}")
        
        if error_info.suggested_actions:
            print("   💡 Sugerencias:")
            for action in error_info.suggested_actions:
                print(f"      • {action}")
        
        if error_info.context:
            print(f"   📝 Contexto: {error_info.context}")
    
    def get_error_history(self, limit: int = 100) -> List[ErrorInfo]:
        """Obtiene historial reciente de errores"""
        return self.error_history[-limit:]

class FileErrorLogger(IErrorLogger):
    """Logger de errores a archivo"""
    
    def __init__(self, log_file: str = "video_processing_errors.log"):
        self.log_file = log_file
        self.error_history: List[ErrorInfo] = []
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Configura logging a archivo"""
        logging.basicConfig(
            filename=self.log_file,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{__name__}_file")
    
    def log_error(self, error_info: ErrorInfo) -> None:
        """Registra error en archivo"""
        self.error_history.append(error_info)
        
        # Crear entrada detallada para archivo
        log_entry = {
            'error_id': error_info.error_id,
            'category': error_info.category.value,
            'severity': error_info.severity.value,
            'message': error_info.message,
            'details': error_info.details,
            'user_message': error_info.user_message,
            'suggested_actions': error_info.suggested_actions,
            'context': error_info.context,
            'timestamp': error_info.timestamp.isoformat()
        }
        
        if error_info.stacktrace:
            log_entry['stacktrace'] = error_info.stacktrace
        
        self.logger.error(f"ERROR_ENTRY: {log_entry}")
    
    def get_error_history(self, limit: int = 100) -> List[ErrorInfo]:
        """Obtiene historial de errores"""
        return self.error_history[-limit:]

class ErrorHandler:
    """Manejador principal de errores del sistema"""
    
    def __init__(self, logger: Optional[IErrorLogger] = None, 
                 notifier: Optional[IErrorNotifier] = None):
        self.logger = logger or ConsoleErrorLogger()
        self.notifier = notifier
        self.error_mappers = self._initialize_error_mappers()
        self._error_counter = 0
    
    def _initialize_error_mappers(self) -> Dict[type, Callable]:
        """Inicializa mappers para diferentes tipos de errores"""
        return {
            FileNotFoundError: self._map_file_not_found,
            PermissionError: self._map_permission_error,
            ConnectionError: self._map_connection_error,
            ValueError: self._map_value_error,
            ImportError: self._map_import_error,
            KeyError: self._map_key_error,
            TypeError: self._map_type_error
        }
    
    def handle_exception(self, exception: Exception, 
                        context: Dict[str, Any] = None) -> ErrorInfo:
        """Maneja una excepción y retorna información del error"""
        
        self._error_counter += 1
        error_id = f"ERR_{self._error_counter:04d}_{int(datetime.now().timestamp())}"
        
        # Si es una excepción de nuestro sistema
        if isinstance(exception, VideoProcessingError):
            error_info = ErrorInfo(
                error_id=error_id,
                category=exception.category,
                severity=exception.severity,
                message=str(exception),
                details=str(exception),
                timestamp=exception.timestamp,
                user_message=self._create_user_message(exception),
                suggested_actions=self._get_suggested_actions(exception),
                context=exception.context,
                stacktrace=traceback.format_exc()
            )
        else:
            # Mapear excepción estándar de Python
            mapper = self.error_mappers.get(type(exception), self._map_generic_error)
            error_info = mapper(exception, error_id, context or {})
        
        # Registrar error
        self.logger.log_error(error_info)
        
        # Notificar si es crítico
        if (error_info.severity == ErrorSeverity.CRITICAL and 
            self.notifier):
            self.notifier.notify_error(error_info)
        
        return error_info
    
    def create_error(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None) -> ErrorInfo:
        """Crea un error manualmente"""
        
        self._error_counter += 1
        error_id = f"ERR_{self._error_counter:04d}_{int(datetime.now().timestamp())}"
        
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            details=message,
            timestamp=datetime.now(),
            user_message=self._create_user_friendly_message(message, category),
            suggested_actions=self._get_category_suggestions(category),
            context=context or {}
        )
        
        self.logger.log_error(error_info)
        return error_info
    
    def _map_file_not_found(self, exception: FileNotFoundError, 
                           error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea FileNotFoundError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            details=f"Archivo no encontrado: {exception.filename}",
            timestamp=datetime.now(),
            user_message="No se pudo encontrar un archivo necesario",
            suggested_actions=[
                "Verificar que el archivo existe",
                "Comprobar la ruta del archivo",
                "Verificar permisos de lectura"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_permission_error(self, exception: PermissionError,
                             error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea PermissionError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            details=f"Permisos insuficientes: {exception.filename}",
            timestamp=datetime.now(),
            user_message="No hay permisos suficientes para acceder al archivo",
            suggested_actions=[
                "Verificar permisos del archivo/directorio",
                "Ejecutar como administrador si es necesario",
                "Cambiar ubicación del archivo"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_connection_error(self, exception: ConnectionError,
                             error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea ConnectionError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            details="Error de conexión de red",
            timestamp=datetime.now(),
            user_message="No se pudo establecer conexión con el servidor",
            suggested_actions=[
                "Verificar conexión a internet",
                "Comprobar configuración de proxy",
                "Intentar más tarde",
                "Verificar URL del servicio"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_value_error(self, exception: ValueError,
                        error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea ValueError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            details="Valor inválido proporcionado",
            timestamp=datetime.now(),
            user_message="Se proporcionó un valor inválido",
            suggested_actions=[
                "Verificar formato de los datos",
                "Comprobar rangos válidos",
                "Revisar documentación"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_import_error(self, exception: ImportError,
                         error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea ImportError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            message=str(exception),
            details=f"Módulo no encontrado: {exception.name}",
            timestamp=datetime.now(),
            user_message="Falta una dependencia necesaria",
            suggested_actions=[
                f"Instalar dependencia: pip install {exception.name}",
                "Verificar requirements.txt",
                "Actualizar entorno virtual"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_key_error(self, exception: KeyError,
                      error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea KeyError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            details=f"Clave no encontrada: {exception.args[0]}",
            timestamp=datetime.now(),
            user_message="Falta configuración requerida",
            suggested_actions=[
                "Verificar archivo de configuración",
                "Comprobar variables de entorno",
                "Revisar documentación de configuración"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_type_error(self, exception: TypeError,
                       error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea TypeError"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            details="Tipo de dato incorrecto",
            timestamp=datetime.now(),
            user_message="Se proporcionó un tipo de dato incorrecto",
            suggested_actions=[
                "Verificar tipos de datos",
                "Comprobar conversiones",
                "Revisar parámetros de función"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _map_generic_error(self, exception: Exception,
                          error_id: str, context: Dict[str, Any]) -> ErrorInfo:
        """Mapea errores genéricos"""
        return ErrorInfo(
            error_id=error_id,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            details=f"Error inesperado: {type(exception).__name__}",
            timestamp=datetime.now(),
            user_message="Ocurrió un error inesperado",
            suggested_actions=[
                "Reintentar la operación",
                "Verificar logs para más detalles",
                "Contactar soporte si persiste"
            ],
            context=context,
            stacktrace=traceback.format_exc()
        )
    
    def _create_user_message(self, exception: VideoProcessingError) -> str:
        """Crea mensaje amigable para usuario"""
        category_messages = {
            ErrorCategory.DOWNLOAD: "Error descargando el video",
            ErrorCategory.AUDIO_PROCESSING: "Error procesando el audio",
            ErrorCategory.TRANSCRIPTION: "Error en la transcripción",
            ErrorCategory.TRANSLATION: "Error en la traducción",
            ErrorCategory.TTS: "Error generando audio",
            ErrorCategory.VIDEO_COMPOSITION: "Error componiendo video final",
            ErrorCategory.FILE_SYSTEM: "Error con archivos",
            ErrorCategory.NETWORK: "Error de conexión",
            ErrorCategory.VALIDATION: "Datos inválidos",
            ErrorCategory.CONFIGURATION: "Error de configuración",
            ErrorCategory.SYSTEM: "Error del sistema"
        }
        
        return category_messages.get(exception.category, "Error inesperado")
    
    def _create_user_friendly_message(self, message: str, category: ErrorCategory) -> str:
        """Crea mensaje amigable desde mensaje técnico"""
        category_prefixes = {
            ErrorCategory.DOWNLOAD: "Descarga:",
            ErrorCategory.AUDIO_PROCESSING: "Audio:",
            ErrorCategory.TRANSCRIPTION: "Transcripción:",
            ErrorCategory.TRANSLATION: "Traducción:",
            ErrorCategory.TTS: "Generación de voz:",
            ErrorCategory.VIDEO_COMPOSITION: "Composición:",
            ErrorCategory.FILE_SYSTEM: "Archivo:",
            ErrorCategory.NETWORK: "Conexión:",
            ErrorCategory.VALIDATION: "Validación:",
            ErrorCategory.CONFIGURATION: "Configuración:",
            ErrorCategory.SYSTEM: "Sistema:"
        }
        
        prefix = category_prefixes.get(category, "Error:")
        return f"{prefix} {message}"
    
    def _get_suggested_actions(self, exception: VideoProcessingError) -> List[str]:
        """Obtiene sugerencias específicas para una excepción"""
        return self._get_category_suggestions(exception.category)
    
    def _get_category_suggestions(self, category: ErrorCategory) -> List[str]:
        """Obtiene sugerencias por categoría"""
        suggestions = {
            ErrorCategory.DOWNLOAD: [
                "Verificar URL del video",
                "Comprobar conexión a internet",
                "Intentar con otro video"
            ],
            ErrorCategory.AUDIO_PROCESSING: [
                "Verificar que el video tenga audio",
                "Comprobar espacio en disco",
                "Reintentar el procesamiento"
            ],
            ErrorCategory.TRANSCRIPTION: [
                "Verificar calidad del audio",
                "Intentar con modelo diferente",
                "Comprobar idioma del audio"
            ],
            ErrorCategory.TRANSLATION: [
                "Verificar API keys",
                "Comprobar conexión a internet",
                "Intentar con traductor diferente"
            ],
            ErrorCategory.TTS: [
                "Verificar texto a convertir",
                "Comprobar configuración de voz",
                "Intentar con otra voz"
            ],
            ErrorCategory.VIDEO_COMPOSITION: [
                "Verificar archivos de video y audio",
                "Comprobar espacio en disco",
                "Validar formatos de archivo"
            ],
            ErrorCategory.FILE_SYSTEM: [
                "Verificar permisos de archivo",
                "Comprobar espacio en disco",
                "Validar ruta del archivo"
            ],
            ErrorCategory.NETWORK: [
                "Verificar conexión a internet",
                "Comprobar configuración de proxy",
                "Intentar más tarde"
            ],
            ErrorCategory.VALIDATION: [
                "Verificar formato de datos",
                "Comprobar valores permitidos",
                "Revisar documentación"
            ],
            ErrorCategory.CONFIGURATION: [
                "Verificar archivo de configuración",
                "Comprobar variables de entorno",
                "Revisar instalación de dependencias"
            ],
            ErrorCategory.SYSTEM: [
                "Reiniciar la aplicación",
                "Verificar recursos del sistema",
                "Contactar soporte técnico"
            ]
        }
        
        return suggestions.get(category, ["Reintentar la operación"])
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de errores"""
        errors = self.logger.get_error_history()
        
        if not errors:
            return {"total_errors": 0}
        
        # Contar por categoría
        category_counts = {}
        severity_counts = {}
        
        for error in errors:
            category = error.category.value
            severity = error.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(errors),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "most_recent": errors[-1] if errors else None,
            "critical_count": severity_counts.get("critical", 0)
        }

# Decorator para manejo automático de errores
def handle_errors(error_handler: ErrorHandler = None, 
                 category: ErrorCategory = ErrorCategory.SYSTEM):
    """Decorator para manejo automático de errores en funciones"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = handler.handle_exception(e, {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Limitar longitud
                    'kwargs': str(kwargs)[:100]
                })
                
                # Re-lanzar como VideoProcessingError
                raise VideoProcessingError(
                    error_info.user_message,
                    category,
                    error_info.severity
                ) from e
        
        return wrapper
    return decorator

# Context manager para manejo de errores
class ErrorContext:
    """Context manager para manejo de errores en bloques de código"""
    
    def __init__(self, error_handler: ErrorHandler = None,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Dict[str, Any] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.category = category
        self.context = context or {}
        self.error_occurred = False
        self.error_info = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.error_info = self.error_handler.handle_exception(
                exc_val, self.context
            )
            
            # Suprimir la excepción original y lanzar VideoProcessingError
            raise VideoProcessingError(
                self.error_info.user_message,
                self.category,
                self.error_info.severity
            ) from exc_val
        
        return False

# Funciones de conveniencia
def create_error_handler(use_file_logging: bool = False, 
                        log_file: str = "video_processing.log") -> ErrorHandler:
    """Crea un manejador de errores con configuración estándar"""
    
    if use_file_logging:
        logger = FileErrorLogger(log_file)
    else:
        logger = ConsoleErrorLogger()
    
    return ErrorHandler(logger)

def handle_critical_error(message: str, context: Dict[str, Any] = None,
                         error_handler: ErrorHandler = None) -> ErrorInfo:
    """Maneja un error crítico y termina la aplicación si es necesario"""
    
    handler = error_handler or ErrorHandler()
    
    error_info = handler.create_error(
        message, 
        ErrorCategory.SYSTEM,
        ErrorSeverity.CRITICAL,
        context
    )
    
    print(f"\n🚨 ERROR CRÍTICO: {message}")
    print("La aplicación no puede continuar.")
    
    return error_info