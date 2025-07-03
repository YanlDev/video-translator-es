"""Interfaces/contratos del sistema"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict 
from typing import Optional, Callable, List, Dict
from .models import ( VideoInfo, DownloadProgress, DownloadResult, AudioExtractionResult,  AudioSeparationResult, AudioProcessingProgress, TranscriptionResult,  SubtitleGenerationResult, LanguageInfo, TranslationResult, LanguagePair,  TranslatorConfig, TextChunk, TranslationProgress, TranslationCostEstimate,  TranslationMetrics, SpanishVoice, SpanishTTSRequest, SpanishTTSResult,  TTSProgress, SpanishVoiceFilter, VideoAsset, AudioTrack, VideoCompositionRequest, VideoCompositionResult, CompositionProgress, CompositionTemplate, ProjectCompositionInfo
)

# INTERFACES PARA DESCARGA DE VIDEOS
class IVideoInfoExtractor(ABC):
    """Interface para extraer información de videos"""
    
    @abstractmethod
    def extract_info(self, url: str) -> Optional[VideoInfo]:
        """Extrae información básica del video"""
        pass

class IVideoDownloader(ABC):
    """Interface para descargar videos"""
    
    @abstractmethod
    def download(self, url: str, output_path: str, 
                progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> DownloadResult:
        """Descarga un video"""
        pass

class IFileNameSanitizer(ABC):
    """Interface para limpiar nombres de archivos"""
    
    @abstractmethod
    def sanitize(self, filename: str, max_length: int = 30) -> str:
        """Limpia y valida nombres de archivos"""
        pass

class IProgressDisplay(ABC):
    """Interface para mostrar progreso"""
    
    @abstractmethod
    def show_progress(self, progress: DownloadProgress) -> None:
        """Muestra el progreso de descarga"""
        pass

# INTERFACES PARA EXTRACCION Y SEPARACION DE AUDIO
class IAudioExtractor(ABC):
    """Interface para extraer audio de videos"""
    
    @abstractmethod
    def extract_audio(self, video_path: str, output_path: str, format: str = "wav", sample_rate: int = 16000, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> AudioExtractionResult:
        """Extrae audio del video en el formato especificado"""
        pass

class IAudioSeparator(ABC):
    """Interface para separar audio en componentes"""
    
    @abstractmethod
    def separate_audio(self, audio_path: str, output_directory: str, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> AudioSeparationResult:
        """Separa audio en voces y acompañamiento"""
        pass

class IAudioQualityAnalyzer(ABC):
    """Interface para analizar calidad de separación"""
    
    @abstractmethod
    def analyze_separation_quality(self, vocals_path: str, accompaniment_path: str) -> float:
        """Analiza y retorna score de calidad (0-1)"""
        pass
    
# INTERFACES PARA TRANSCRIPCION Y SUBTITULOS    
class ITranscriber(ABC):
    """Interface para transcribir audio a texto"""
    
    @abstractmethod
    def transcribe_audio(self, audio_path: str, force_language: Optional[str] = None, model_size: str = "base") -> TranscriptionResult:
        """Transcribe audio a texto con detección automática de idioma"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Retorna lista de códigos de idiomas soportados"""
        pass

class ISubtitleGenerator(ABC):
    """Interface para generar archivos de subtítulos"""
    
    @abstractmethod
    def generate_srt(self, transcription: TranscriptionResult, output_path: str) -> SubtitleGenerationResult:
        """Genera archivo SRT desde transcripción"""
        pass
    
    @abstractmethod
    def generate_vtt(self, transcription: TranscriptionResult, output_path: str) -> SubtitleGenerationResult:
        """Genera archivo VTT desde transcripción"""
        pass

class ILanguageDetector(ABC):
    """Interface para detectar idioma de audio"""
    
    @abstractmethod
    def detect_language(self, audio_path: str) -> LanguageInfo:
        """Detecta el idioma del audio"""
        pass


# ===== NUEVAS INTERFACES PARA TRADUCCIÓN =====

class ITranslator(ABC):
    """Interface base para todos los traductores"""
    
    @abstractmethod
    def translate_text(self, text: str, source_language: str, target_language: str = "es") -> TranslationResult:
        """
        Traduce texto de un idioma a otro
        
        Args:
            text: Texto a traducir
            source_language: Código del idioma origen ('en', 'fr', 'de', etc.)
            target_language: Código del idioma destino (por defecto 'es' para español)
            
        Returns:
            TranslationResult con el resultado de la traducción
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Retorna lista de códigos de idiomas soportados
        
        Returns:
            Lista de códigos de idioma ('en', 'es', 'fr', etc.)
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str, source_language: str, target_language: str = "es") -> TranslationCostEstimate:
        """
        Estima el costo de traducir un texto
        
        Args:
            text: Texto a evaluar
            source_language: Idioma origen
            target_language: Idioma destino
            
        Returns:
            TranslationCostEstimate con la estimación de costo
        """
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """
        Verifica si el traductor está correctamente configurado
        
        Returns:
            True si está listo para usar, False si necesita configuración
        """
        pass
    
    @abstractmethod
    def get_translator_name(self) -> str:
        """
        Retorna el nombre del traductor
        
        Returns:
            Nombre del traductor ('openai', 'google', 'deepl')
        """
        pass

class ILongTextTranslator(ABC):
    """Interface para traductores que manejan textos largos con chunking"""
    
    @abstractmethod
    def translate_long_text(self, text: str, source_language: str, target_language: str = "es", max_chunk_size: int = 5000, progress_callback: Optional[Callable[[TranslationProgress], None]] = None) -> TranslationResult:
        """
        Traduce textos largos dividiéndolos en chunks automáticamente
        
        Args:
            text: Texto largo a traducir
            source_language: Idioma origen
            target_language: Idioma destino
            max_chunk_size: Tamaño máximo por chunk en caracteres
            progress_callback: Función para reportar progreso
            
        Returns:
            TranslationResult con el texto completo traducido
        """
        pass
    
    @abstractmethod
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 5000) -> List[TextChunk]:
        """
        Divide texto largo en chunks inteligentemente
        
        Args:
            text: Texto a dividir
            max_chunk_size: Tamaño máximo por chunk
            
        Returns:
            Lista de TextChunk
        """
        pass

class ITranslationQualityAnalyzer(ABC):
    """Interface para analizar calidad de traducciones"""
    
    @abstractmethod
    def analyze_translation_quality(self, original_text: str, translated_text: str, language_pair: LanguagePair) -> TranslationMetrics:
        """
        Analiza la calidad de una traducción
        
        Args:
            original_text: Texto original
            translated_text: Texto traducido
            language_pair: Par de idiomas usado
            
        Returns:
            TranslationMetrics con métricas de calidad
        """
        pass
    
    @abstractmethod
    def detect_translation_issues(self, original_text: str, translated_text: str) -> List[str]:
        """
        Detecta problemas potenciales en la traducción
        
        Args:
            original_text: Texto original
            translated_text: Texto traducido
            
        Returns:
            Lista de problemas detectados
        """
        pass

class ILanguageDetector(ABC):
    """Interface para detectores de idioma (ya existía en transcripción, la extendemos)"""
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """
        Detecta el idioma de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código del idioma detectado
        """
        pass
    
    @abstractmethod
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """
        Retorna confianza de detección para múltiples idiomas
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con idiomas y scores de confianza
        """
        pass

class ITranslationCache(ABC):
    """Interface para caché de traducciones (evitar retraducciones)"""
    
    @abstractmethod
    def get_cached_translation(self, text_hash: str, language_pair: LanguagePair, translator: str) -> Optional[TranslationResult]:
        """
        Busca traducción en caché
        
        Args:
            text_hash: Hash del texto original
            language_pair: Par de idiomas
            translator: Nombre del traductor usado
            
        Returns:
            TranslationResult si existe en caché, None si no
        """
        pass
    
    @abstractmethod
    def cache_translation(self, text_hash: str, language_pair: LanguagePair, translator: str, result: TranslationResult) -> bool:
        """
        Guarda traducción en caché
        
        Args:
            text_hash: Hash del texto original
            language_pair: Par de idiomas
            translator: Nombre del traductor
            result: Resultado a cachear
            
        Returns:
            True si se guardó exitosamente
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> bool:
        """Limpia todo el caché de traducciones"""
        pass

class ITranslationConfigValidator(ABC):
    """Interface para validar configuraciones de traductores"""
    
    @abstractmethod
    def validate_openai_config(self, api_key: str, model: str = "gpt-4") -> bool:
        """
        Valida configuración de OpenAI
        
        Args:
            api_key: API key de OpenAI
            model: Modelo a usar
            
        Returns:
            True si la configuración es válida
        """
        pass
    
    @abstractmethod
    def validate_deepl_config(self, api_key: str) -> bool:
        """
        Valida configuración de DeepL
        
        Args:
            api_key: API key de DeepL
            
        Returns:
            True si la configuración es válida
        """
        pass
    
    @abstractmethod
    def test_translator_connection(self, translator: ITranslator) -> bool:
        """
        Prueba conexión con un traductor
        
        Args:
            translator: Instancia del traductor a probar
            
        Returns:
            True si la conexión es exitosa
        """
        pass

class ITranslationFormatter(ABC):
    """Interface para formatear traducciones manteniendo estructura"""
    
    @abstractmethod
    def preserve_timestamps(self, original_segments: List, translated_text: str) -> List:
        """
        Preserva timestamps al aplicar traducción a segmentos
        
        Args:
            original_segments: Segmentos originales con timestamps
            translated_text: Texto traducido completo
            
        Returns:
            Segmentos con timestamps y texto traducido
        """
        pass
    
    @abstractmethod
    def format_for_subtitles(self, translation_result: TranslationResult, max_chars_per_line: int = 42) -> List[str]:
        """
        Formatea traducción para subtítulos
        
        Args:
            translation_result: Resultado de traducción
            max_chars_per_line: Máximo caracteres por línea
            
        Returns:
            Lista de líneas formateadas para subtítulos
        """
        pass
    
    

# INTERFACES PARA TEXT-TO-SPEECH EN ESPAÑOL 


class ISpanishVoiceManager(ABC):
    """Interface para gestionar voces disponibles en español"""
    
    @abstractmethod
    def get_available_voices(self) -> List[SpanishVoice]:
        """Retorna lista de voces españolas disponibles"""
        pass
    
    @abstractmethod
    def filter_voices(self, filter_criteria: SpanishVoiceFilter) -> List[SpanishVoice]:
        """Filtra voces según criterios específicos"""
        pass
    
    @abstractmethod
    def get_recommended_voice(self, gender: str = "Female") -> Optional[SpanishVoice]:
        """Retorna voz recomendada por defecto"""
        pass
    
    @abstractmethod
    def get_voice_by_id(self, voice_id: str) -> Optional[SpanishVoice]:
        """Busca voz específica por ID"""
        pass

class ISpanishTTSProvider(ABC):
    """Interface base para proveedores de TTS en español"""
    
    @abstractmethod
    def generate_speech(self, request: SpanishTTSRequest, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """
        Genera audio en español desde texto
        
        Args:
            request: Solicitud de TTS con texto y configuración
            progress_callback: Callback para reportar progreso
            
        Returns:
            SpanishTTSResult con el resultado de la generación
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el proveedor está disponible y configurado"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Retorna nombre del proveedor (edge, google, azure)"""
        pass
    
    @abstractmethod
    def get_supported_voices(self) -> List[SpanishVoice]:
        """Retorna voces españolas soportadas por este proveedor"""
        pass
    
    @abstractmethod
    def validate_text(self, text: str) -> bool:
        """Valida que el texto sea apropiado para TTS"""
        pass

class ISpanishTTSProcessor(ABC):
    """Interface para procesamiento completo de TTS en español"""
    
    @abstractmethod
    def process_spanish_text(self, text: str, output_path: str, voice_preference: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """
        Procesa texto completo en español y genera audio
        
        Args:
            text: Texto en español a convertir
            output_path: Ruta donde guardar el audio
            voice_preference: Preferencia de voz ("Male", "Female" o ID específico)
            progress_callback: Callback para progreso
            
        Returns:
            SpanishTTSResult con el resultado completo
        """
        pass
    
    @abstractmethod
    def process_long_text(self, text: str, output_path: str, max_chunk_size: int = 1000, voice_preference: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """
        Procesa textos largos dividiéndolos en chunks
        
        Args:
            text: Texto largo en español
            output_path: Ruta de salida
            max_chunk_size: Tamaño máximo por chunk
            voice_preference: Preferencia de voz
            progress_callback: Callback para progreso
            
        Returns:
            SpanishTTSResult con audio combinado
        """
        pass
    
    @abstractmethod
    def get_available_providers(self) -> List[str]:
        """Retorna lista de proveedores disponibles"""
        pass

class ISpanishTTSQualityAnalyzer(ABC):
    """Interface para analizar calidad del audio TTS generado"""
    
    @abstractmethod
    def analyze_audio_quality(self, audio_path: str) -> float:
        """
        Analiza calidad del audio TTS
        
        Args:
            audio_path: Ruta del archivo de audio
            
        Returns:
            Score de calidad 0.0-1.0
        """
        pass
    
    @abstractmethod
    def detect_audio_issues(self, audio_path: str) -> List[str]:
        """
        Detecta problemas en el audio generado
        
        Args:
            audio_path: Ruta del archivo de audio
            
        Returns:
            Lista de problemas detectados
        """
        pass
    
    @abstractmethod
    def validate_audio_sync(self, original_text: str, audio_path: str) -> bool:
        """
        Valida que el audio corresponda al texto original
        
        Args:
            original_text: Texto original
            audio_path: Audio generado
            
        Returns:
            True si el audio es consistente con el texto
        """
        pass

class ISpanishTTSCache(ABC):
    """Interface para caché de audio TTS generado"""
    
    @abstractmethod
    def get_cached_audio(self, text_hash: str, voice_id: str) -> Optional[str]:
        """
        Busca audio en caché
        
        Args:
            text_hash: Hash del texto
            voice_id: ID de la voz usada
            
        Returns:
            Ruta del archivo de audio si existe en caché
        """
        pass
    
    @abstractmethod
    def cache_audio(self, text_hash: str, voice_id: str, audio_path: str) -> bool:
        """
        Guarda audio en caché
        
        Args:
            text_hash: Hash del texto
            voice_id: ID de la voz
            audio_path: Ruta del audio a cachear
            
        Returns:
            True si se guardó exitosamente
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> bool:
        """Limpia todo el caché de TTS"""
        pass

class ISpanishTTSConfigValidator(ABC):
    """Interface para validar configuraciones de TTS"""
    
    @abstractmethod
    def validate_edge_tts_config(self) -> bool:
        """
        Valida configuración de Edge TTS
        
        Returns:
            True si Edge TTS está disponible
        """
        pass
    
    @abstractmethod
    def validate_google_tts_config(self, credentials_path: Optional[str] = None) -> bool:
        """
        Valida configuración de Google TTS
        
        Args:
            credentials_path: Ruta a credenciales de Google Cloud
            
        Returns:
            True si Google TTS está configurado correctamente
        """
        pass
    
    @abstractmethod
    def validate_azure_tts_config(self, api_key: Optional[str] = None) -> bool:
        """
        Valida configuración de Azure TTS
        
        Args:
            api_key: API key de Azure
            
        Returns:
            True si Azure TTS está configurado
        """
        pass
    
    @abstractmethod
    def test_provider_connection(self, provider: ISpanishTTSProvider) -> bool:
        """
        Prueba conexión con un proveedor TTS
        
        Args:
            provider: Instancia del proveedor a probar
            
        Returns:
            True si la conexión es exitosa
        """
        pass

class ISpanishTextPreprocessor(ABC):
    """Interface para preprocesamiento de texto antes de TTS"""
    
    @abstractmethod
    def clean_text_for_tts(self, text: str) -> str:
        """
        Limpia texto para optimizar TTS
        
        Args:
            text: Texto original en español
            
        Returns:
            Texto limpio y optimizado para TTS
        """
        pass
    
    @abstractmethod
    def normalize_spanish_text(self, text: str) -> str:
        """
        Normaliza texto en español (números, abreviaciones, etc.)
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        pass
    
    @abstractmethod
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divide texto en oraciones apropiadas para TTS
        
        Args:
            text: Texto completo
            
        Returns:
            Lista de oraciones
        """
        pass
    
    @abstractmethod
    def estimate_audio_duration(self, text: str, voice: SpanishVoice) -> float:
        """
        Estima duración del audio que se generará
        
        Args:
            text: Texto a analizar
            voice: Voz que se usará
            
        Returns:
            Duración estimada en segundos
        """
        pass

class ISpanishAudioPostprocessor(ABC):
    """Interface para postprocesamiento del audio TTS generado"""
    
    @abstractmethod
    def normalize_audio_volume(self, audio_path: str, target_volume: float = 0.8) -> bool:
        """
        Normaliza volumen del audio
        
        Args:
            audio_path: Ruta del archivo de audio
            target_volume: Volumen objetivo (0.0-1.0)
            
        Returns:
            True si se normalizó exitosamente
        """
        pass
    
    @abstractmethod
    def apply_audio_effects(self, audio_path: str, effects: List[str]) -> bool:
        """
        Aplica efectos al audio (reverb, compressor, etc.)
        
        Args:
            audio_path: Ruta del archivo
            effects: Lista de efectos a aplicar
            
        Returns:
            True si se aplicaron los efectos
        """
        pass
    
    @abstractmethod
    def convert_audio_format(self, input_path: str, output_path: str, target_format: str = "wav") -> bool:
        """
        Convierte audio a formato específico
        
        Args:
            input_path: Archivo de entrada
            output_path: Archivo de salida
            target_format: Formato objetivo
            
        Returns:
            True si la conversión fue exitosa
        """
        pass

# Interface adaptativa para manejo inteligente de múltiples proveedores
class IAdaptiveSpanishTTS(ABC):
    """Interface para TTS adaptativo que maneja múltiples proveedores"""
    
    @abstractmethod
    def generate_with_best_available(self, text: str, output_path: str, voice_preference: Optional[str] = None) -> SpanishTTSResult:
        """
        Genera audio usando el mejor proveedor disponible
        
        Args:
            text: Texto en español
            output_path: Ruta de salida
            voice_preference: Preferencia de voz
            
        Returns:
            SpanishTTSResult usando el mejor proveedor disponible
        """
        pass
    
    @abstractmethod
    def get_provider_rankings(self) -> List[str]:
        """
        Retorna proveedores ordenados por calidad/disponibilidad
        
        Returns:
            Lista de proveedores en orden de preferencia
        """
        pass
    
    @abstractmethod
    def fallback_to_next_provider(self, failed_provider: str) -> Optional[ISpanishTTSProvider]:
        """
        Cambia al siguiente proveedor disponible
        
        Args:
            failed_provider: Proveedor que falló
            
        Returns:
            Siguiente proveedor disponible o None
        """
        pass
    
# AGREGA ESTAS INTERFACES AL FINAL DE interfaces.py
# (después de las interfaces TTS existentes)

# INTERFACES PARA COMPOSICIÓN DE VIDEO

class IVideoAnalyzer(ABC):
    """Interface para análisis de archivos de video"""
    
    @abstractmethod
    def analyze_video(self, video_path: str) -> 'VideoAsset':
        """Analiza archivo de video y retorna sus propiedades"""
        pass
    
    @abstractmethod
    def get_video_duration(self, video_path: str) -> float:
        """Obtiene duración del video en segundos"""
        pass
    
    @abstractmethod
    def get_video_resolution(self, video_path: str) -> str:
        """Obtiene resolución del video (ej: "1920x1080")"""
        pass
    
    @abstractmethod
    def has_audio_track(self, video_path: str) -> bool:
        """Verifica si el video tiene pista de audio"""
        pass

class IAudioMixer(ABC):
    """Interface para mezcla de pistas de audio"""
    
    @abstractmethod
    def mix_audio_tracks(self, audio_tracks: List['AudioTrack'], output_path: str,
                        duration: float,
                        progress_callback: Optional[Callable[['CompositionProgress'], None]] = None) -> bool:
        """Mezcla múltiples pistas de audio en una sola"""
        pass
    
    @abstractmethod
    def apply_ducking(self, voice_track: str, background_track: str, output_path: str,
                     ducking_threshold: float = -20.0, ducking_ratio: float = 0.3) -> bool:
        """Aplica ducking (reduce música cuando hay voz)"""
        pass
    
    @abstractmethod
    def normalize_audio_levels(self, audio_path: str, target_level: float = -16.0) -> bool:
        """Normaliza niveles de audio a un target específico"""
        pass

class IVideoComposer(ABC):
    """Interface principal para composición de video"""
    
    @abstractmethod
    def compose_video(self, request: 'VideoCompositionRequest',
                     progress_callback: Optional[Callable[['CompositionProgress'], None]] = None) -> 'VideoCompositionResult':
        """Compone video final combinando todos los elementos"""
        pass
    
    @abstractmethod
    def validate_composition_request(self, request: 'VideoCompositionRequest') -> List[str]:
        """Valida que la solicitud de composición sea válida"""
        pass
    
    @abstractmethod
    def estimate_processing_time(self, request: 'VideoCompositionRequest') -> float:
        """Estima tiempo de procesamiento en segundos"""
        pass

class IVideoEncoder(ABC):
    """Interface para codificación de video"""
    
    @abstractmethod
    def encode_video(self, input_video: str, input_audio: str, output_path: str,
                    quality: str = "high") -> bool:
        """Codifica video final con nueva pista de audio"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Retorna formatos de salida soportados"""
        pass
    
    @abstractmethod
    def optimize_for_size(self, video_path: str, target_size_mb: float) -> bool:
        """Optimiza video para tamaño específico"""
        pass

class IProjectCompositionManager(ABC):
    """Interface para gestión de composición desde proyectos"""
    
    @abstractmethod
    def create_project_composition(self, project_path: str) -> Optional['ProjectCompositionInfo']:
        """Crea información de composición desde estructura de proyecto"""
        pass
    
    @abstractmethod
    def auto_detect_assets(self, project_path: str) -> Dict[str, Optional[str]]:
        """Detecta automáticamente assets disponibles en el proyecto"""
        pass
    
    @abstractmethod
    def validate_project_for_composition(self, project_path: str) -> List[str]:
        """Valida que el proyecto tenga todo lo necesario para composición"""
        pass

class IAdaptiveVideoComposer(ABC):
    """Interface para compositor adaptativo con múltiples motores"""
    
    @abstractmethod
    def compose_with_best_available(self, request: 'VideoCompositionRequest') -> 'VideoCompositionResult':
        """Compone usando el mejor motor disponible"""
        pass
    
    @abstractmethod
    def get_available_composers(self) -> List[str]:
        """Retorna lista de compositores disponibles"""
        pass
    
    @abstractmethod
    def fallback_to_next_composer(self, failed_composer: str) -> Optional['IVideoComposer']:
        """Cambia al siguiente compositor si uno falla"""
        pass