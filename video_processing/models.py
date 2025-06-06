"""Modelos de datos para el sistema de descarga"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

#MODELOS PARA DESCARGA DE VIDEOS
@dataclass
class VideoInfo:
    """Información del video"""
    titulo: str
    duracion: int
    canal: str
    id: str
    url: str
    
    def get_duration_formatted(self) -> str:
        """Formatea la duración de manera legible"""
        if not self.duracion:
            return "Desconocida"
        
        horas = self.duracion // 3600
        minutos = (self.duracion % 3600) // 60
        segs = self.duracion % 60
        
        if horas > 0:
            return f"{int(horas):02d}:{int(minutos):02d}:{int(segs):02d}"
        else:
            return f"{int(minutos):02d}:{int(segs):02d}"

@dataclass
class DownloadProgress:
    """Progreso de descarga"""
    percentage: float
    downloaded_bytes: int
    total_bytes: Optional[int]
    status: str
    message: str = "Descargando"

@dataclass
class DownloadResult:
    """Resultado de descarga"""
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    video_info: Optional[VideoInfo] = None
    
    
#MODELO PARA EXTRACCION Y SEPARACION DE AUDIO
@dataclass
class AudioExtractionResult:
    """Resultado de extracción de audio"""
    success: bool
    audio_file_path: Optional[str] = None
    original_video_path: Optional[str] = None
    audio_format: str = "wav"
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    file_size_mb: float = 0.0
    error_message: Optional[str] = None

@dataclass
class AudioSeparationResult:
    """Resultado de separación de audio"""
    success: bool
    vocals_path: Optional[str] = None
    accompaniment_path: Optional[str] = None
    original_audio_path: Optional[str] = None
    separation_method: str = ""
    processing_time_seconds: float = 0.0
    quality_score: float = 0.0  # 0-1, estimación de calidad
    error_message: Optional[str] = None

@dataclass
class AudioProcessingProgress:
    """Progreso de procesamiento de audio"""
    stage: str  # "extracting", "separating", "analyzing"
    percentage: float
    current_step: str
    estimated_time_remaining: float = 0.0
    
    
#MODELOS PARA TRANSCRIPCIÓN MULTIIDIOMA
@dataclass
class TranscriptionSegment:
    """Segmento de transcripción con timestamps"""
    start_time: float  # Segundos
    end_time: float    # Segundos
    text: str
    confidence: float = 0.0
    
    def get_duration(self) -> float:
        """Duración del segmento en segundos"""
        return self.end_time - self.start_time
    
    def format_timestamp_srt(self, seconds: float) -> str:
        """Convierte segundos a formato SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def to_srt_format(self, segment_number: int) -> str:
        """Convierte el segmento a formato SRT"""
        start_formatted = self.format_timestamp_srt(self.start_time)
        end_formatted = self.format_timestamp_srt(self.end_time)
        
        return f"{segment_number}\n{start_formatted} --> {end_formatted}\n{self.text}\n"

@dataclass
class LanguageInfo:
    """Información del idioma detectado"""
    code: str           # 'es', 'en', 'fr', etc.
    name: str           # 'Spanish', 'English', 'French'
    confidence: float   # 0.0 - 1.0
    is_spanish: bool    # True si ya está en español
    
    @staticmethod
    def from_whisper_code(language_code: str, confidence: float = 0.0) -> 'LanguageInfo':
        """Crea LanguageInfo desde código de Whisper"""
        language_names = {
            'es': 'Spanish', 'en': 'English', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'th': 'Thai', 'vi': 'Vietnamese', 'nl': 'Dutch', 'sv': 'Swedish',
            'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'pl': 'Polish',
            'tr': 'Turkish', 'he': 'Hebrew', 'cs': 'Czech', 'sk': 'Slovak',
            'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian',
            'sr': 'Serbian', 'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian',
            'lt': 'Lithuanian', 'uk': 'Ukrainian', 'be': 'Belarusian', 'mk': 'Macedonian',
            'sq': 'Albanian', 'eu': 'Basque', 'gl': 'Galician', 'ca': 'Catalan'
        }
        
        return LanguageInfo(
            code=language_code,
            name=language_names.get(language_code, f"Unknown ({language_code})"),
            confidence=confidence,
            is_spanish=(language_code == 'es')
        )

@dataclass
class TranscriptionResult:
    """Resultado completo de transcripción multiidioma"""
    success: bool
    text: str = ""
    language: Optional[LanguageInfo] = None
    segments: List[TranscriptionSegment] = None
    model_used: str = ""
    processing_time: float = 0.0
    audio_duration: float = 0.0
    word_count: int = 0
    needs_translation: bool = True
    confidence_average: float = 0.0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.segments is None:
            self.segments = []
        
        # Calcular estadísticas automáticamente
        if self.text:
            self.word_count = len(self.text.split())
        
        if self.segments:
            confidences = [seg.confidence for seg in self.segments if seg.confidence > 0]
            if confidences:
                self.confidence_average = sum(confidences) / len(confidences)
        
        # Determinar si necesita traducción
        if self.language:
            self.needs_translation = not self.language.is_spanish

@dataclass
class SubtitleGenerationResult:
    """Resultado de generación de subtítulos"""
    success: bool
    srt_file_path: Optional[str] = None
    segment_count: int = 0
    total_duration: float = 0.0
    error_message: Optional[str] = None

#MODELOS PARA TRADUCCIÓN MULTIIDIOMA
@dataclass
class LanguagePair:
    """Par de idiomas para traducción"""
    source_language: str  # Código del idioma origen (ej: 'en', 'fr', 'de')
    target_language: str  # Código del idioma destino (ej: 'es')
    source_name: str      # Nombre del idioma origen (ej: 'English', 'French')
    target_name: str      # Nombre del idioma destino (ej: 'Spanish')
    
    def __str__(self) -> str:
        return f"{self.source_name} → {self.target_name}"
    
    def get_direction_code(self) -> str:
        """Retorna código de dirección (ej: 'en-es')"""
        return f"{self.source_language}-{self.target_language}"
    
    @staticmethod
    def create_to_spanish(source_lang: str, source_name: str) -> 'LanguagePair':
        """Crea par de idioma hacia español"""
        return LanguagePair(
            source_language=source_lang,
            target_language='es',
            source_name=source_name,
            target_name='Spanish'
        )

@dataclass
class TextChunk:
    """Fragmento de texto para traducir (para textos largos)"""
    chunk_id: int
    text: str
    start_position: int  # Posición en el texto original
    end_position: int    # Posición final en el texto original
    word_count: int
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())

@dataclass
class TranslationResult:
    """Resultado de una traducción"""
    success: bool
    original_text: str = ""
    translated_text: str = ""
    language_pair: Optional[LanguagePair] = None
    translator_used: str = ""  # 'openai', 'google', 'deepl'
    processing_time: float = 0.0
    character_count: int = 0
    word_count: int = 0
    confidence_score: float = 0.0  # 0.0 - 1.0
    cost_estimate: float = 0.0  # Estimación de costo en USD
    chunks_processed: int = 1  # Número de fragmentos procesados
    error_message: Optional[str] = None
    
    def __post_init__(self):
        # Calcular estadísticas automáticamente
        if self.original_text:
            self.character_count = len(self.original_text)
            self.word_count = len(self.original_text.split())

@dataclass
class TranslationMetrics:
    """Métricas de calidad de traducción"""
    translation_id: str
    length_preservation: float  # Ratio de longitud original vs traducida
    terminology_consistency: float  # Consistencia en términos técnicos
    fluency_score: float  # Fluidez estimada
    adequacy_score: float  # Adecuación del contenido
    overall_quality: float  # Score general (0.0 - 1.0)
    
    def get_quality_description(self) -> str:
        """Retorna descripción de calidad"""
        if self.overall_quality >= 0.9:
            return "Excelente"
        elif self.overall_quality >= 0.8:
            return "Muy buena"
        elif self.overall_quality >= 0.7:
            return "Buena"
        elif self.overall_quality >= 0.6:
            return "Aceptable"
        else:
            return "Necesita revisión"

@dataclass
class TranslatorConfig:
    """Configuración para un traductor específico"""
    translator_name: str
    api_key: Optional[str] = None
    model_name: Optional[str] = None  # Para OpenAI: 'gpt-4', 'gpt-3.5-turbo'
    max_chars_per_request: int = 5000  # Límite por petición
    timeout_seconds: int = 30
    preserve_formatting: bool = True
    custom_prompt: Optional[str] = None  # Para OpenAI
    
    def is_configured(self) -> bool:
        """Verifica si el traductor está configurado correctamente"""
        if self.translator_name.lower() == 'google':
            return True  # Google no requiere API key
        else:
            return self.api_key is not None and len(self.api_key.strip()) > 0

@dataclass
class TranslationProgress:
    """Progreso de traducción (para textos largos)"""
    total_chunks: int
    current_chunk: int
    completed_chars: int
    total_chars: int
    current_translator: str
    estimated_time_remaining: float = 0.0
    
    def get_percentage(self) -> float:
        """Retorna porcentaje completado"""
        if self.total_chars == 0:
            return 0.0
        return (self.completed_chars / self.total_chars) * 100

@dataclass
class TranslationCostEstimate:
    """Estimación de costos de traducción"""
    service_name: str
    character_count: int
    estimated_cost_usd: float
    currency: str = "USD"
    rate_per_char: float = 0.0
    includes_tax: bool = False
    
    def format_cost(self) -> str:
        """Formatea el costo para mostrar"""
        if self.estimated_cost_usd == 0.0:
            return "Gratis"
        elif self.estimated_cost_usd < 0.01:
            return f"< $0.01 {self.currency}"
        else:
            return f"${self.estimated_cost_usd:.3f} {self.currency}"

@dataclass
class TranslationBatch:
    """Lote de traducciones para procesamiento en batch"""
    batch_id: str
    chunks: List[TextChunk]
    language_pair: LanguagePair
    translator_config: TranslatorConfig
    created_at: datetime
    priority: int = 1  # 1=alta, 2=media, 3=baja
    
    def get_total_characters(self) -> int:
        """Retorna total de caracteres en el lote"""
        return sum(len(chunk.text) for chunk in self.chunks)
    
    def get_total_words(self) -> int:
        """Retorna total de palabras en el lote"""
        return sum(chunk.word_count for chunk in self.chunks)

@dataclass
class TranslationSessionInfo:
    """Información de sesión de traducción completa"""
    session_id: str
    video_title: str
    original_language: str
    target_language: str
    total_duration_seconds: float
    character_count: int
    word_count: int
    translator_used: str
    processing_start_time: datetime
    processing_end_time: Optional[datetime] = None
    total_cost: float = 0.0
    success: bool = False
    
    def get_processing_duration(self) -> float:
        """Retorna duración del procesamiento en segundos"""
        if not self.processing_end_time:
            return 0.0
        delta = self.processing_end_time - self.processing_start_time
        return delta.total_seconds()
    
    def get_words_per_minute(self) -> float:
        """Retorna palabras por minuto procesadas"""
        duration = self.get_processing_duration()
        if duration == 0:
            return 0.0
        return (self.word_count / duration) * 60

# MODELOS PARA TEXT-TO-SPEECH (TTS) - AGREGAR A models.py

@dataclass
class SpanishVoice:
    """Información de una voz TTS en español"""
    id: str                    # Identificador único de la voz
    name: str                  # Nombre descriptivo
    locale: str                # Locale específico (es-ES, es-MX, es-AR)
    gender: str                # Male, Female
    provider: str              # edge, google, azure
    is_neural: bool            # Si es voz neural (mejor calidad)
    sample_rate: int = 22050   # Sample rate en Hz
    
    def __str__(self) -> str:
        return f"{self.name} ({self.locale}) - {self.gender}"
    
    def get_quality_score(self) -> int:
        """Retorna score de calidad 1-3"""
        if self.is_neural:
            return 3  # Alta calidad
        else:
            return 2  # Calidad estándar

@dataclass
class SpanishTTSRequest:
    """Solicitud de generación TTS en español"""
    text: str                  # Texto en español a convertir
    voice: SpanishVoice        # Voz española a utilizar
    output_path: str           # Ruta donde guardar el audio
    speed: float = 1.0         # Velocidad (0.5 - 2.0)
    volume: float = 1.0        # Volumen (0.0 - 1.0)
    
    def get_estimated_duration(self) -> float:
        """Estima duración del audio en segundos"""
        # Aproximación: ~150 palabras por minuto en español
        words = len(self.text.split())
        duration_minutes = words / 150
        return (duration_minutes * 60) / self.speed

@dataclass
class SpanishTTSResult:
    """Resultado de generación TTS en español"""
    success: bool
    audio_file_path: Optional[str] = None
    duration_seconds: float = 0.0
    file_size_mb: float = 0.0
    voice_used: Optional[SpanishVoice] = None
    processing_time: float = 0.0
    character_count: int = 0
    error_message: Optional[str] = None
    
    def get_audio_info(self) -> Dict[str, Any]:
        """Retorna información del audio generado"""
        return {
            'duration': f"{self.duration_seconds:.1f}s",
            'size': f"{self.file_size_mb:.1f}MB",
            'sample_rate': f"{self.actual_sample_rate}Hz",
            'voice': str(self.voice_used) if self.voice_used else "Unknown",
            'processing_time': f"{self.processing_time:.1f}s"
        }

@dataclass
class TTSProgress:
    """Progreso de generación TTS"""
    stage: str                 # "preparing", "generating", "saving", "completed"
    percentage: float          # 0.0 - 100.0
    current_chunk: int         # Chunk actual (para textos largos)
    total_chunks: int          # Total de chunks
    estimated_time_remaining: float = 0.0
    current_text_preview: str = ""  # Muestra del texto actual
    
    def get_stage_description(self) -> str:
        """Descripción legible del stage"""
        descriptions = {
            "preparing": "Preparando texto y voz",
            "generating": "Generando audio con IA",
            "saving": "Guardando archivo de audio",
            "completed": "Generación completada"
        }
        return descriptions.get(self.stage, self.stage)

@dataclass
class SpanishVoiceFilter:
    """Filtros para búsqueda de voces españolas"""
    locale: Optional[str] = None        # es-ES, es-MX, es-AR, etc.
    gender: Optional[str] = None        # Male, Female
    provider: Optional[str] = None      # edge, google, azure
    neural_only: bool = True            # Solo voces neurales por defecto
    
    def matches(self, voice: SpanishVoice) -> bool:
        """Verifica si una voz coincide con los filtros"""
        if self.locale and voice.locale != self.locale:
            return False
        if self.gender and voice.gender != self.gender:
            return False
        if self.provider and voice.provider != self.provider:
            return False
        if self.neural_only and not voice.is_neural:
            return False
        
        return True

@dataclass
class TTSConfiguration:
    """Configuración del sistema TTS"""
    preferred_provider: str = "edge"    # Proveedor preferido
    fallback_providers: List[str] = None  # Proveedores de fallback
    default_voice_filters: VoiceFilter = None  # Filtros por defecto
    chunk_size: int = 1000             # Tamaño de chunk para textos largos
    max_text_length: int = 10000       # Máximo texto por request
    default_format: str = "wav"        # Formato por defecto
    default_sample_rate: int = 22050   # Sample rate por defecto
    enable_ssml: bool = True           # Habilitar SSML si está disponible
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = ["google", "azure", "edge"]
        if self.default_voice_filters is None:
            self.default_voice_filters = VoiceFilter(
                language="es",
                neural_only=True,
                min_quality_score=3
            )

@dataclass
class TTSBatch:
    """Lote de generaciones TTS para procesamiento masivo"""
    batch_id: str
    requests: List[TTSRequest]
    created_at: datetime
    priority: int = 1  # 1=alta, 2=media, 3=baja
    
    def get_total_characters(self) -> int:
        """Total de caracteres en el lote"""
        return sum(len(req.text) for req in self.requests)
    
    def get_estimated_duration(self) -> float:
        """Duración estimada total del lote"""
        return sum(req.get_estimated_duration() for req in self.requests)
    
    def get_estimated_processing_time(self) -> float:
        """Tiempo estimado de procesamiento"""
        # Estimación: ~10x tiempo real para generación
        return self.get_estimated_duration() * 10

@dataclass
class TTSSessionInfo:
    """Información de sesión TTS completa"""
    session_id: str
    project_name: str
    original_text: str
    voice_used: Voice
    total_duration_seconds: float
    character_count: int
    word_count: int
    processing_start_time: datetime
    processing_end_time: Optional[datetime] = None
    total_cost: float = 0.0
    success: bool = False
    
    def get_processing_duration(self) -> float:
        """Duración del procesamiento en segundos"""
        if not self.processing_end_time:
            return 0.0
        delta = self.processing_end_time - self.processing_start_time
        return delta.total_seconds()
    
    def get_characters_per_second(self) -> float:
        """Caracteres procesados por segundo"""
        duration = self.get_processing_duration()
        if duration == 0:
            return 0.0
        return self.character_count / duration
    
    def get_real_time_factor(self) -> float:
        """Factor de tiempo real (1.0 = tiempo real)"""
        processing_time = self.get_processing_duration()
        if processing_time == 0 or self.total_duration_seconds == 0:
            return 0.0
        return processing_time / self.total_duration_seconds

@dataclass
class SSMLConfig:
    """Configuración para SSML (Speech Synthesis Markup Language)"""
    enable_breaks: bool = True          # Pausas automáticas
    enable_emphasis: bool = True        # Énfasis en palabras importantes
    enable_prosody: bool = True         # Control de prosodia
    sentence_break_time: str = "0.5s"   # Pausa entre oraciones
    paragraph_break_time: str = "1.0s"  # Pausa entre párrafos
    emphasis_level: str = "moderate"    # none, reduced, moderate, strong
    
    def wrap_text_with_ssml(self, text: str) -> str:
        """Envuelve texto con marcas SSML"""
        if not any([self.enable_breaks, self.enable_emphasis, self.enable_prosody]):
            return text
        
        # Implementación básica de SSML
        ssml_text = f'<speak version="1.0" xml:lang="es">'
        
        # Dividir en párrafos y oraciones
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if i > 0 and self.enable_breaks:
                ssml_text += f'<break time="{self.paragraph_break_time}"/>'
            
            sentences = paragraph.split('. ')
            for j, sentence in enumerate(sentences):
                if j > 0 and self.enable_breaks:
                    ssml_text += f'<break time="{self.sentence_break_time}"/>'
                
                # Agregar énfasis si está habilitado
                if self.enable_emphasis and any(word.isupper() for word in sentence.split()):
                    # Simple emphasis para palabras en mayúsculas
                    words = sentence.split()
                    processed_words = []
                    for word in words:
                        if word.isupper() and len(word) > 2:
                            processed_words.append(f'<emphasis level="{self.emphasis_level}">{word.lower()}</emphasis>')
                        else:
                            processed_words.append(word)
                    sentence = ' '.join(processed_words)
                
                ssml_text += sentence
            
            if not paragraph.endswith('.'):
                ssml_text += '.'
        
        ssml_text += '</speak>'
        return ssml_text