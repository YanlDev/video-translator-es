"""Factories para crear configuraciones"""

# Imports existentes
from .services import VideoDownloadService, AudioProcessingService, TranscriptionService
from .downloaders import (YouTubeDownloader, YouTubeInfoExtractor, SimpleFileNameSanitizer, ConsoleProgressDisplay)
from .audio_processors import (MoviePyAudioExtractor, SpleeterCLISeparator, BasicFallbackSeparator)
from .transcribers import WhisperMultilingualTranscriber, SRTSubtitleGenerator
from .interfaces import IVideoDownloader, IVideoInfoExtractor

# Imports para traducción y TTS
from typing import Optional
import os

# =============================================================================
# FACTORIES EXISTENTES
# =============================================================================

class VideoDownloaderFactory:
    """Factory para crear diferentes configuraciones de downloader"""
    
    @staticmethod
    def create_youtube_downloader() -> VideoDownloadService:
        """Crea downloader configurado para YouTube"""
        file_sanitizer = SimpleFileNameSanitizer()
        progress_display = ConsoleProgressDisplay()
        info_extractor = YouTubeInfoExtractor()
        
        downloader = YouTubeDownloader(file_sanitizer, progress_display)
        
        return VideoDownloadService(downloader, info_extractor)
    
    @staticmethod
    def create_custom_downloader(downloader: IVideoDownloader, 
                                info_extractor: IVideoInfoExtractor) -> VideoDownloadService:
        """Crea downloader con implementaciones personalizadas"""
        return VideoDownloadService(downloader, info_extractor)

class AudioProcessingFactory:
    """Factory para crear procesadores de audio limpios"""
    
    @staticmethod
    def create_spleeter_processor() -> AudioProcessingService:
        """Crea procesador con Spleeter + fallback FFmpeg"""
        
        audio_extractor = MoviePyAudioExtractor()
        primary_separator = SpleeterCLISeparator(model_name="2stems-16kHz")
        fallback_separator = BasicFallbackSeparator()
        
        return AudioProcessingService(
            audio_extractor=audio_extractor,
            primary_separator=primary_separator,
            fallback_separator=fallback_separator
        )
    
    @staticmethod
    def create_basic_processor() -> AudioProcessingService:
        """Crea procesador básico (solo FFmpeg)"""
        
        audio_extractor = MoviePyAudioExtractor()
        separator = BasicFallbackSeparator()
        
        return AudioProcessingService(
            audio_extractor=audio_extractor,
            primary_separator=separator,
            fallback_separator=separator
        )

class TranscriptionFactory:
    """Factory para crear servicios de transcripción multiidioma"""
    
    @staticmethod
    def create_whisper_transcription_service() -> TranscriptionService:
        """Crea servicio de transcripción con faster-whisper + subtítulos"""
        
        transcriber = WhisperMultilingualTranscriber()
        subtitle_generator = SRTSubtitleGenerator()
        
        return TranscriptionService(transcriber, subtitle_generator)
    
    @staticmethod
    def create_fast_transcription_service() -> TranscriptionService:
        """Crea servicio de transcripción rápida (modelo tiny)"""
        
        transcriber = WhisperMultilingualTranscriber()
        subtitle_generator = SRTSubtitleGenerator()
        
        return TranscriptionService(transcriber, subtitle_generator)
    
    @staticmethod
    def create_high_quality_transcription_service() -> TranscriptionService:
        """Crea servicio de transcripción de alta calidad (modelo large)"""
        
        transcriber = WhisperMultilingualTranscriber()
        subtitle_generator = SRTSubtitleGenerator()
        
        return TranscriptionService(transcriber, subtitle_generator)

# =============================================================================
# TRANSLATION FACTORIES
# =============================================================================

class TranslationFactory:
    """Factory para crear servicios de traducción multiidioma"""
    
    @staticmethod
    def create_openai_translation_service(api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Crea servicio de traducción con OpenAI como traductor principal"""
        from .services import TranslationService
        from .translators import OpenAITranslator, BasicTranslationQualityAnalyzer
        
        primary_translator = OpenAITranslator(api_key=api_key, model=model)
        quality_analyzer = BasicTranslationQualityAnalyzer()
        
        return TranslationService(
            primary_translator=primary_translator,
            quality_analyzer=quality_analyzer
        )
    
    @staticmethod
    def create_google_translation_service():
        """Crea servicio de traducción con Google Translate como traductor principal"""
        from .services import TranslationService
        from .translators import GoogleTranslator, BasicTranslationQualityAnalyzer
        
        primary_translator = GoogleTranslator()
        quality_analyzer = BasicTranslationQualityAnalyzer()
        
        return TranslationService(
            primary_translator=primary_translator,
            quality_analyzer=quality_analyzer
        )
    
    @staticmethod
    def create_adaptive_translation_service():
        """Crea servicio adaptativo que detecta automáticamente los traductores disponibles"""
        from .services import AdaptiveTranslationService
        
        return AdaptiveTranslationService()
    
    @staticmethod
    def create_best_available_service():
        """Crea el mejor servicio de traducción disponible según configuración actual"""
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                return TranslationFactory.create_openai_translation_service(openai_key)
            except:
                pass
        
        return TranslationFactory.create_google_translation_service()

class TranslationConfigurationFactory:
    """Factory para crear configuraciones específicas de traducción"""
    
    @staticmethod
    def create_high_quality_config():
        """Configuración para máxima calidad (OpenAI GPT-4o)"""
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                return TranslationFactory.create_openai_translation_service(
                    api_key=openai_key,
                    model="gpt-4o"
                )
            except:
                pass
        
        return TranslationFactory.create_google_translation_service()
    
    @staticmethod
    def create_fast_config():
        """Configuración para máxima velocidad (Google Translate)"""
        return TranslationFactory.create_google_translation_service()
    
    @staticmethod
    def create_cost_effective_config():
        """Configuración para mínimo costo (Google gratis)"""
        return TranslationFactory.create_google_translation_service()

# =============================================================================
# TTS FACTORIES
# =============================================================================

class TTSServiceFactory:
    """Factory para crear servicios de TTS en español"""
    
    @staticmethod
    def create_spanish_tts_service():
        """Crea servicio TTS en español con Edge como proveedor principal"""
        from .services import SpanishTTSService
        from .tts_generators import SpanishTTSProcessor, SpanishVoiceManager
        
        tts_processor = SpanishTTSProcessor()
        voice_manager = SpanishVoiceManager()
        
        return SpanishTTSService(
            tts_processor=tts_processor,
            voice_manager=voice_manager
        )
    
    @staticmethod
    def create_adaptive_tts_service():
        """Crea servicio TTS adaptativo"""
        from .services import SpanishTTSService
        from .tts_generators import AdaptiveSpanishTTS, SpanishVoiceManager
        
        adaptive_processor = AdaptiveSpanishTTS()
        voice_manager = SpanishVoiceManager()
        
        # Wrapper simple para compatibilidad
        class AdaptiveTTSWrapper:
            def __init__(self, adaptive_tts):
                self.adaptive_tts = adaptive_tts
            
            def process_spanish_text(self, text, output_path, voice_preference=None, progress_callback=None):
                return self.adaptive_tts.generate_with_best_available(text, output_path, voice_preference)
            
            def process_long_text(self, text, output_path, max_chunk_size=1000, voice_preference=None, progress_callback=None):
                return self.adaptive_tts.generate_with_best_available(text, output_path, voice_preference)
            
            def get_available_providers(self):
                return self.adaptive_tts.get_provider_rankings()
        
        adaptive_wrapper = AdaptiveTTSWrapper(adaptive_processor)
        
        return SpanishTTSService(
            tts_processor=adaptive_wrapper,
            voice_manager=voice_manager
        )

class TTSConfigurationFactory:
    """Factory para crear configuraciones específicas de TTS"""
    
    @staticmethod
    def create_high_quality_tts():
        """Configuración TTS para máxima calidad"""
        return TTSServiceFactory.create_spanish_tts_service()
    
    @staticmethod
    def create_fast_tts():
        """Configuración TTS para máxima velocidad"""
        return TTSServiceFactory.create_spanish_tts_service()

# =============================================================================
# INTEGRATED FACTORIES
# =============================================================================

class IntegratedTranslationFactory:
    """Factory para crear servicios integrados completos"""
    
    @staticmethod
    def create_transcription_translation_service(transcription_service, translation_config: str = "adaptive"):
        """Crea servicio integrado de transcripción + traducción"""
        
        if translation_config == "adaptive":
            translation_service = TranslationFactory.create_adaptive_translation_service()
        elif translation_config == "high_quality":
            translation_service = TranslationConfigurationFactory.create_high_quality_config()
        elif translation_config == "fast":
            translation_service = TranslationConfigurationFactory.create_fast_config()
        else:
            translation_service = TranslationFactory.create_adaptive_translation_service()
        
        from .services import TranscriptionTranslationService
        
        return TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )

class CompleteVideoTranslationFactory:
    """Factory maestro para pipelines completos con TTS"""
    
    @staticmethod
    def create_complete_translator_pipeline():
        """Crea pipeline completo: descarga + audio + transcripción + traducción + TTS"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_whisper_transcription_service()
        translation_service = TranslationFactory.create_adaptive_translation_service()
        tts_service = TTSServiceFactory.create_spanish_tts_service()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_service
        }
    
    @staticmethod
    def create_quality_focused_translator():
        """Pipeline optimizado para máxima calidad"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_high_quality_transcription_service()
        translation_service = TranslationConfigurationFactory.create_high_quality_config()
        tts_service = TTSConfigurationFactory.create_high_quality_tts()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_service
        }
    
    @staticmethod
    def create_fast_translator():
        """Pipeline optimizado para velocidad"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_basic_processor()
        transcription_service = TranscriptionFactory.create_fast_transcription_service()
        translation_service = TranslationConfigurationFactory.create_fast_config()
        tts_service = TTSConfigurationFactory.create_fast_tts()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_service
        }

# =============================================================================
# VIDEO COMPOSITION FACTORIES
# =============================================================================

class VideoCompositionFactory:
    """Factory para crear servicios de composición de video"""
    
    @staticmethod
    def create_video_composition_service():
        """Crea servicio de composición con FFmpeg como compositor principal"""
        from .services import VideoCompositionService
        from .video_composers import AdaptiveVideoComposer, ProjectCompositionManager
        
        video_composer = AdaptiveVideoComposer()
        project_manager = ProjectCompositionManager()
        
        return VideoCompositionService(
            video_composer=video_composer,
            project_manager=project_manager
        )
    
    @staticmethod
    def create_ffmpeg_composition_service():
        """Crea servicio de composición específicamente con FFmpeg"""
        from .services import VideoCompositionService
        from .video_composers import FFmpegVideoComposer, ProjectCompositionManager
        
        ffmpeg_composer = FFmpegVideoComposer()
        project_manager = ProjectCompositionManager()
        
        return VideoCompositionService(
            video_composer=ffmpeg_composer,
            project_manager=project_manager
        )

class VideoCompositionConfigurationFactory:
    """Factory para crear configuraciones específicas de composición"""
    
    @staticmethod
    def create_high_quality_composition():
        """Configuración para máxima calidad de composición"""
        return VideoCompositionFactory.create_video_composition_service()
    
    @staticmethod
    def create_fast_composition():
        """Configuración para composición rápida"""
        return VideoCompositionFactory.create_video_composition_service()
    
    @staticmethod
    def create_standard_composition():
        """Configuración estándar de composición"""
        return VideoCompositionFactory.create_video_composition_service()

# =============================================================================
# INTEGRATED FACTORIES - ACTUALIZADAS CON COMPOSICIÓN
# =============================================================================

class CompleteVideoProcessingFactory:
    """Factory maestro para pipelines completos incluyendo composición final"""
    
    @staticmethod
    def create_complete_video_processing_service():
        """Crea servicio completo: descarga + audio + transcripción + traducción + TTS + composición"""
        from .services import CompleteVideoProcessingService
        
        # Servicios base
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_whisper_transcription_service()
        translation_service = TranslationFactory.create_adaptive_translation_service()
        tts_service = TTSServiceFactory.create_spanish_tts_service()
        composition_service = VideoCompositionFactory.create_video_composition_service()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        # Servicio integrado transcripción + traducción
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        # Servicio completo hasta TTS
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        # Servicio completo incluyendo composición final
        return CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )
    
    @staticmethod
    def create_quality_focused_processing():
        """Pipeline completo optimizado para máxima calidad"""
        from .services import CompleteVideoProcessingService
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_high_quality_transcription_service()
        translation_service = TranslationConfigurationFactory.create_high_quality_config()
        tts_service = TTSConfigurationFactory.create_high_quality_tts()
        composition_service = VideoCompositionConfigurationFactory.create_high_quality_composition()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        return CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )
    
    @staticmethod
    def create_fast_processing():
        """Pipeline completo optimizado para velocidad"""
        from .services import CompleteVideoProcessingService
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_basic_processor()
        transcription_service = TranscriptionFactory.create_fast_transcription_service()
        translation_service = TranslationConfigurationFactory.create_fast_config()
        tts_service = TTSConfigurationFactory.create_fast_tts()
        composition_service = VideoCompositionConfigurationFactory.create_fast_composition()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        return CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )

# ACTUALIZAR CompleteVideoTranslationFactory PARA INCLUIR COMPOSICIÓN

class CompleteVideoTranslationFactory:
    """Factory que crea pipelines completos con traducción + TTS + composición"""
    
    @staticmethod
    def create_complete_translator_pipeline():
        """Crea pipeline completo: descarga + audio + transcripción + traducción + TTS + composición"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_whisper_transcription_service()
        translation_service = TranslationFactory.create_adaptive_translation_service()
        tts_service = TTSServiceFactory.create_spanish_tts_service()
        composition_service = VideoCompositionFactory.create_video_composition_service()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService, CompleteVideoProcessingService
        
        # Servicios integrados paso a paso
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        complete_processing_service = CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'composition_service': composition_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_video_to_spanish_service,
            'full_pipeline_service': complete_processing_service
        }
    
    @staticmethod
    def create_quality_focused_translator():
        """Pipeline optimizado para máxima calidad (incluyendo composición)"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_spleeter_processor()
        transcription_service = TranscriptionFactory.create_high_quality_transcription_service()
        translation_service = TranslationConfigurationFactory.create_high_quality_config()
        tts_service = TTSConfigurationFactory.create_high_quality_tts()
        composition_service = VideoCompositionConfigurationFactory.create_high_quality_composition()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService, CompleteVideoProcessingService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        complete_processing_service = CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'composition_service': composition_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_video_to_spanish_service,
            'full_pipeline_service': complete_processing_service
        }
    
    @staticmethod
    def create_fast_translator():
        """Pipeline optimizado para velocidad (incluyendo composición)"""
        
        download_service = VideoDownloaderFactory.create_youtube_downloader()
        audio_service = AudioProcessingFactory.create_basic_processor()
        transcription_service = TranscriptionFactory.create_fast_transcription_service()
        translation_service = TranslationConfigurationFactory.create_fast_config()
        tts_service = TTSConfigurationFactory.create_fast_tts()
        composition_service = VideoCompositionConfigurationFactory.create_fast_composition()
        
        from .services import TranscriptionTranslationService, CompleteVideoToSpanishService, CompleteVideoProcessingService
        
        transcription_translation_service = TranscriptionTranslationService(
            transcription_service=transcription_service,
            translation_service=translation_service
        )
        
        complete_video_to_spanish_service = CompleteVideoToSpanishService(
            transcription_translation_service=transcription_translation_service,
            tts_service=tts_service
        )
        
        complete_processing_service = CompleteVideoProcessingService(
            complete_video_to_spanish_service=complete_video_to_spanish_service,
            video_composition_service=composition_service
        )
        
        return {
            'download_service': download_service,
            'audio_service': audio_service,
            'transcription_service': transcription_service,
            'translation_service': translation_service,
            'tts_service': tts_service,
            'composition_service': composition_service,
            'integrated_service': transcription_translation_service,
            'complete_service': complete_video_to_spanish_service,
            'full_pipeline_service': complete_processing_service
        }

# =============================================================================
# FUNCIONES DE CONVENIENCIA ACTUALIZADAS
# =============================================================================

def create_video_translator(config_type: str = "adaptive"):
    """
    Función de conveniencia para crear un traductor de videos completo con TTS y composición
    
    Args:
        config_type: 'adaptive', 'quality', 'fast'
        
    Returns:
        Dict con todos los servicios configurados incluyendo composición final
    """
    if config_type == "quality":
        return CompleteVideoTranslationFactory.create_quality_focused_translator()
    elif config_type == "fast":
        return CompleteVideoTranslationFactory.create_fast_translator()
    else:
        return CompleteVideoTranslationFactory.create_complete_translator_pipeline()

def create_complete_video_processor(config_type: str = "standard"):
    """
    Función de conveniencia para crear procesador completo de videos
    
    Args:
        config_type: 'standard', 'quality', 'fast'
        
    Returns:
        CompleteVideoProcessingService configurado
    """
    if config_type == "quality":
        return CompleteVideoProcessingFactory.create_quality_focused_processing()
    elif config_type == "fast":
        return CompleteVideoProcessingFactory.create_fast_processing()
    else:
        return CompleteVideoProcessingFactory.create_complete_video_processing_service()

def create_video_composition_only():
    """Función de conveniencia para crear solo servicio de composición"""
    return VideoCompositionFactory.create_video_composition_service()

def process_video_url_to_spanish(video_url: str, config_type: str = "adaptive", voice_gender: str = "Female"):
    """
    Función de conveniencia para procesar URL completa hasta video final
    
    Args:
        video_url: URL del video a procesar
        config_type: Configuración del pipeline
        voice_gender: Género de voz TTS
        
    Returns:
        Resultado completo del procesamiento
    """
    processor = create_complete_video_processor(config_type)
    return processor.process_video_url_to_final_spanish_video(video_url, voice_preference=voice_gender)