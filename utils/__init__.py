"""Utilidades compartidas del sistema"""

from .file_manager import (
    FileManager, DirectoryManager, ProjectStructureManager,
    ProjectPaths, FileInfo
)
from .progress_tracker import (
    ProgressTracker, ProgressDisplay, ConsoleProgressDisplay, SilentProgressDisplay,
    PipelineProgressTracker, ProgressInfo, ProgressContext,
    create_simple_progress_tracker, create_silent_progress_tracker, create_pipeline_tracker
)
from .error_handler import (
    ErrorHandler, VideoProcessingError, ValidationError, NetworkError,
    FileSystemError, ConfigurationError, ErrorInfo, ErrorSeverity, ErrorCategory,
    IErrorLogger, ConsoleErrorLogger, FileErrorLogger, ErrorContext,
    handle_errors, create_error_handler, handle_critical_error
)
from .validators import (
    URLValidator, FileValidator, ConfigValidator, ProjectValidator,
    VideoProcessingRequestValidator, VideoCompositionRequestValidator,
    CompositeValidator, ValidationResult, IValidator,
    validate_video_url, validate_file_exists, validate_project_structure,
    validate_config_dict, validate_operation_context, create_validator_suite
)
from .config_manager import (
    ConfigManager, EnvironmentConfig, SystemConfig,
    IConfigLoader, JSONConfigLoader, YAMLConfigLoader,
    get_default_config_manager, load_config_from_file, create_config_from_template,
    validate_environment_setup, TemporaryConfig
)

__all__ = [
    # Gestión de archivos
    'FileManager',
    'DirectoryManager', 
    'ProjectStructureManager',
    'ProjectPaths',
    'FileInfo',
    
    # Progreso y tracking
    'ProgressTracker',
    'ProgressDisplay',
    'ConsoleProgressDisplay',
    'SilentProgressDisplay',
    'PipelineProgressTracker',
    'ProgressInfo',
    'ProgressContext',
    'create_simple_progress_tracker',
    'create_silent_progress_tracker',
    'create_pipeline_tracker',
    
    # Manejo de errores
    'ErrorHandler',
    'VideoProcessingError',
    'ValidationError',
    'NetworkError',
    'FileSystemError',
    'ConfigurationError',
    'ErrorInfo',
    'ErrorSeverity',
    'ErrorCategory',
    'IErrorLogger',
    'ConsoleErrorLogger',
    'FileErrorLogger',
    'ErrorContext',
    'handle_errors',
    'create_error_handler',
    'handle_critical_error',
    
    # Validaciones
    'URLValidator',
    'FileValidator',
    'ConfigValidator',
    'ProjectValidator',
    'VideoProcessingRequestValidator',
    'VideoCompositionRequestValidator',
    'CompositeValidator',
    'ValidationResult',
    'IValidator',
    'validate_video_url',
    'validate_file_exists',
    'validate_project_structure',
    'validate_config_dict',
    'validate_operation_context',
    'create_validator_suite',
    
    # Configuración
    'ConfigManager',
    'EnvironmentConfig',
    'SystemConfig',
    'IConfigLoader',
    'JSONConfigLoader',
    'YAMLConfigLoader',
    'get_default_config_manager',
    'load_config_from_file',
    'create_config_from_template',
    'validate_environment_setup',
    'TemporaryConfig'
]

__version__ = "1.0.0"
__description__ = "Utilidades compartidas para procesamiento de videos siguiendo principios SOLID"

# Información sobre la arquitectura
__architecture_info__ = {
    "design_patterns": [
        "Dependency Injection",
        "Factory Pattern", 
        "Strategy Pattern",
        "Observer Pattern",
        "Command Pattern"
    ],
    "solid_principles": {
        "S": "Single Responsibility - Cada clase tiene una responsabilidad específica",
        "O": "Open/Closed - Abierto para extensión, cerrado para modificación",
        "L": "Liskov Substitution - Las implementaciones son intercambiables",
        "I": "Interface Segregation - Interfaces específicas y cohesivas",
        "D": "Dependency Inversion - Dependencias de abstracciones, no concreciones"
    },
    "key_benefits": [
        "Código mantenible y extensible",
        "Fácil testing con mocks",
        "Bajo acoplamiento entre componentes",
        "Alta cohesión dentro de módulos",
        "Configuración centralizada"
    ]
}