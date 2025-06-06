"""Servicios principales del sistema"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

# Imports existentes para download y audio
from .interfaces import (IVideoDownloader, IVideoInfoExtractor, IAudioExtractor, IAudioSeparator, ITranscriber, ISubtitleGenerator, ITranslator, ILongTextTranslator, ITranslationQualityAnalyzer,ISpanishTTSProvider, ISpanishTTSProcessor, ISpanishVoiceManager,IVideoComposer, IProjectCompositionManager)
from .models import (DownloadResult, DownloadProgress, AudioProcessingProgress,TranscriptionResult, SubtitleGenerationResult, TranslationResult, LanguagePair, TranslationProgress, 
TranslationMetrics, TranslationSessionInfo, SpanishTTSResult, TTSProgress, SpanishVoice,VideoCompositionRequest, VideoCompositionResult, CompositionProgress)
from .translators import (OpenAITranslator, GoogleTranslator, DeepLTranslator, BasicTranslationQualityAnalyzer)

# SERVICIOS EXISTENTES (VideoDownloadService, AudioProcessingService)

class VideoDownloadService:
    """Servicio principal para descarga de videos"""
    
    def __init__(self, downloader: IVideoDownloader, info_extractor: IVideoInfoExtractor):
        """Constructor con inyección de dependencias"""
        self.downloader = downloader
        self.info_extractor = info_extractor
    
    def download_video(self, url: str, output_directory: str, progress_callback: Optional[Callable[[DownloadProgress], None]] = None, ask_confirmation: bool = True) -> DownloadResult:
        """Descarga un video completo con validaciones"""
        
        # Validar URL
        if not self._is_valid_youtube_url(url):
            return DownloadResult(
                success=False,
                error_message="URL de YouTube no válida"
            )
        
        # Crear directorio
        os.makedirs(output_directory, exist_ok=True)
        
        # Extraer información
        print("🔍 Obteniendo información del video...")
        video_info = self.info_extractor.extract_info(url)
        
        if not video_info:
            return DownloadResult(
                success=False,
                error_message="No se pudo obtener información del video"
            )
        
        # Mostrar información
        self._display_video_info(video_info)
        
        # Confirmar descarga (opcional)
        if ask_confirmation and not self._confirm_download():
            return DownloadResult(
                success=False,
                error_message="Descarga cancelada por el usuario"
            )
        
        # Descargar
        print("\n🚀 Iniciando descarga...")
        return self.downloader.download(url, output_directory, progress_callback)
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Valida si es una URL de YouTube"""
        valid_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        return any(domain in url.lower() for domain in valid_domains)
    
    def _display_video_info(self, video_info) -> None:
        """Muestra información del video"""
        print("\n📊 INFORMACIÓN DEL VIDEO")
        print("-" * 40)
        print(f"🎬 Título: {video_info.titulo}")
        print(f"👤 Canal: {video_info.canal}")
        print(f"⏱️  Duración: {video_info.get_duration_formatted()}")
        print(f"🆔 ID: {video_info.id}")
        print()
    
    def _confirm_download(self) -> bool:
        """Confirma si el usuario quiere continuar"""
        response = input("¿Continuar con la descarga? (s/n): ").lower().strip()
        return response in ['s', 'si', 'sí', 'y', 'yes']

class AudioProcessingService:
    """Servicio para procesamiento completo de audio con fallback"""
    
    def __init__(self, audio_extractor: IAudioExtractor, primary_separator: IAudioSeparator, fallback_separator: IAudioSeparator):
        """Constructor con inyección de dependencias"""
        self.audio_extractor = audio_extractor
        self.primary_separator = primary_separator
        self.fallback_separator = fallback_separator
    
    def process_video_audio(self, video_path: str, output_directory: str, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> dict:
        """Procesa video completo: extracción + separación con fallback"""
        
        results = {
            'extraction': None,
            'separation': None,
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        # Paso 1: Extraer audio
        print("🎵 Paso 1: Extrayendo audio del video...")
        audio_output_path = os.path.join(output_directory, "extracted_audio.wav")
        
        extraction_result = self.audio_extractor.extract_audio(
            video_path, audio_output_path, progress_callback=progress_callback
        )
        
        results['extraction'] = extraction_result
        
        if not extraction_result.success:
            results['total_time'] = time.time() - start_time
            return results
        
        # Paso 2: Separar audio con sistema de fallback
        print("🎵 Paso 2: Separando voces y música...")
        
        # Intentar con separador primario (Spleeter)
        separation_result = self.primary_separator.separate_audio(
            audio_output_path, output_directory, progress_callback=progress_callback
        )
        
        # Si falla, usar fallback (FFmpeg básico)
        if not separation_result.success:
            print("⚠️  Separador primario falló, usando fallback básico...")
            separation_result = self.fallback_separator.separate_audio(
                audio_output_path, output_directory, progress_callback=progress_callback
            )
        
        results['separation'] = separation_result
        results['total_time'] = time.time() - start_time
        
        # Resumen final
        if separation_result.success:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO")
            print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
            print(f"🤖 Método usado: {separation_result.separation_method}")
            print(f"📊 Calidad estimada: {separation_result.quality_score:.1%}")
            print(f"📁 Archivos generados:")
            print(f"   🎤 {os.path.basename(separation_result.vocals_path)}")
            print(f"   🎵 {os.path.basename(separation_result.accompaniment_path)}")
        else:
            print(f"❌ Error en procesamiento de audio")
        
        return results

# NUEVO SERVICIO: TRANSCRIPCIÓN MULTIIDIOMA

class TranscriptionService:
    """Servicio para transcripción completa multiidioma"""
    
    def __init__(self, transcriber: ITranscriber, subtitle_generator: ISubtitleGenerator):
        """Constructor con inyección de dependencias"""
        self.transcriber = transcriber
        self.subtitle_generator = subtitle_generator
    
    def process_audio_transcription(self, audio_path: str, output_directory: str, force_language: Optional[str] = None, model_size: str = "base", generate_subtitles: bool = True) -> dict:
        """Procesa transcripción completa con subtítulos opcionales"""
        
        results = {
            'transcription': None,
            'subtitles_srt': None,
            'subtitles_vtt': None,
            'text_file': None,
            'metadata_file': None,
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        # Crear directorio de salida
        os.makedirs(output_directory, exist_ok=True)
        
        # Paso 1: Transcribir audio
        print("🎤 Transcribiendo audio multiidioma...")
        
        transcription_result = self.transcriber.transcribe_audio(
            audio_path=audio_path,
            force_language=force_language,
            model_size=model_size
        )
        
        results['transcription'] = transcription_result
        
        if not transcription_result.success:
            results['total_time'] = time.time() - start_time
            print(f"❌ Error en transcripción: {transcription_result.error_message}")
            return results
        
        # Paso 2: Guardar transcripción en texto plano
        base_name = Path(audio_path).stem
        text_output = os.path.join(output_directory, f"{base_name}_transcription.txt")
        
        try:
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(transcription_result.text)
            results['text_file'] = text_output
            print(f"💾 Texto guardado: {os.path.basename(text_output)}")
        except Exception as e:
            print(f"⚠️  Error guardando texto: {e}")
        
        # Paso 3: Guardar metadata completa
        metadata_output = os.path.join(output_directory, f"{base_name}_metadata.json")
        
        try:
            metadata = {
                'language': {
                    'code': transcription_result.language.code,
                    'name': transcription_result.language.name,
                    'confidence': transcription_result.language.confidence,
                    'is_spanish': transcription_result.language.is_spanish
                },
                'needs_translation': transcription_result.needs_translation,
                'text': transcription_result.text,
                'word_count': transcription_result.word_count,
                'model_used': transcription_result.model_used,
                'processing_time': transcription_result.processing_time,
                'audio_duration': transcription_result.audio_duration,
                'confidence_average': transcription_result.confidence_average,
                'segment_count': len(transcription_result.segments),
                'timestamp': time.time()
            }
            
            with open(metadata_output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            results['metadata_file'] = metadata_output
            print(f"💾 Metadata guardada: {os.path.basename(metadata_output)}")
            
        except Exception as e:
            print(f"⚠️  Error guardando metadata: {e}")
        
        # Paso 4: Generar subtítulos (si se solicita)
        if generate_subtitles and transcription_result.segments:
            print("📄 Generando subtítulos...")
            
            # Generar SRT
            try:
                srt_output = os.path.join(output_directory, f"{base_name}_subtitles.srt")
                srt_result = self.subtitle_generator.generate_srt(transcription_result, srt_output)
                results['subtitles_srt'] = srt_result
                
                if srt_result.success:
                    print(f"💾 SRT generado: {os.path.basename(srt_output)}")
                else:
                    print(f"⚠️  Error generando SRT: {srt_result.error_message}")
                    
            except Exception as e:
                print(f"⚠️  Error en generación SRT: {e}")
            
            # Generar VTT
            try:
                vtt_output = os.path.join(output_directory, f"{base_name}_subtitles.vtt")
                vtt_result = self.subtitle_generator.generate_vtt(transcription_result, vtt_output)
                results['subtitles_vtt'] = vtt_result
                
                if vtt_result.success:
                    print(f"💾 VTT generado: {os.path.basename(vtt_output)}")
                else:
                    print(f"⚠️  Error generando VTT: {vtt_result.error_message}")
                    
            except Exception as e:
                print(f"⚠️  Error en generación VTT: {e}")
        
        results['total_time'] = time.time() - start_time
        
        # Paso 5: Resumen final
        self._display_transcription_summary(transcription_result, results)
        
        return results
    
    def _display_transcription_summary(self, transcription: TranscriptionResult, results: dict):
        """Muestra resumen de la transcripción"""
        print(f"\n🎉 TRANSCRIPCIÓN COMPLETADA")
        print("=" * 50)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        print(f"🌍 Idioma detectado: {transcription.language.name} ({transcription.language.code})")
        print(f"📊 Confianza idioma: {transcription.language.confidence:.1%}")
        print(f"📝 Palabras transcritas: {transcription.word_count}")
        print(f"🎬 Segmentos con timestamps: {len(transcription.segments)}")
        print(f"⏱️  Duración audio: {transcription.audio_duration:.1f}s")
        print(f"🤖 Modelo usado: {transcription.model_used}")
        
        # Estado de traducción
        if transcription.needs_translation:
            print(f"🔄 Necesita traducción: {transcription.language.name} → Español")
        else:
            print(f"🇪🇸 Ya está en español - no necesita traducción")
        
        # Archivos generados
        print(f"\n📁 Archivos generados:")
        if results['text_file']:
            print(f"   📄 {os.path.basename(results['text_file'])}")
        if results['metadata_file']:
            print(f"   📊 {os.path.basename(results['metadata_file'])}")
        if results['subtitles_srt'] and results['subtitles_srt'].success:
            print(f"   🎬 {os.path.basename(results['subtitles_srt'].srt_file_path)}")
        if results['subtitles_vtt'] and results['subtitles_vtt'].success:
            print(f"   🎬 {os.path.basename(results['subtitles_vtt'].srt_file_path)}")
        
        # Muestra de texto
        if transcription.text:
            sample_length = min(150, len(transcription.text))
            sample_text = transcription.text[:sample_length]
            if len(transcription.text) > sample_length:
                sample_text += "..."
            print(f"\n💬 Muestra del texto transcrito:")
            print(f"   \"{sample_text}\"")

# SERVICIO INTEGRADO: VIDEO COMPLETO CON TRANSCRIPCIÓN

class VideoProcessingCompleteService:
    """Servicio que integra descarga + audio + transcripción"""
    
    def __init__(self, download_service: VideoDownloadService, audio_service: AudioProcessingService, transcription_service: TranscriptionService):
        """Constructor con todos los servicios"""
        self.download_service = download_service
        self.audio_service = audio_service
        self.transcription_service = transcription_service
    
    def process_video_complete(self, url: str, output_directory: str, transcribe_vocals_only: bool = True, transcription_model: str = "base", force_language: Optional[str] = None) -> dict:
        """Procesa video completo: descarga → audio → separación → transcripción"""
        
        complete_results = {
            'download': None,
            'audio_processing': None,
            'transcription': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        print("🚀 INICIANDO PROCESAMIENTO COMPLETO DE VIDEO")
        print("=" * 60)
        
        try:
            # Paso 1: Descargar video
            print("📥 Paso 1/3: Descargando video...")
            download_result = self.download_service.download_video(
                url, output_directory, ask_confirmation=False
            )
            complete_results['download'] = download_result
            
            if not download_result.success:
                print(f"❌ Error en descarga: {download_result.error_message}")
                return complete_results
            
            # Paso 2: Procesar audio (extracción + separación)
            print("\n🎵 Paso 2/3: Procesando audio...")
            audio_results = self.audio_service.process_video_audio(
                download_result.file_path, output_directory
            )
            complete_results['audio_processing'] = audio_results
            
            if not audio_results['separation'] or not audio_results['separation'].success:
                print("❌ Error en procesamiento de audio")
                return complete_results
            
            # Paso 3: Transcribir audio
            print("\n🎤 Paso 3/3: Transcribiendo audio...")
            
            # Decidir qué audio transcribir
            if transcribe_vocals_only and audio_results['separation'].vocals_path:
                audio_to_transcribe = audio_results['separation'].vocals_path
                print("🎤 Transcribiendo solo las voces (mayor precisión)")
            else:
                audio_to_transcribe = audio_results['extraction'].audio_file_path
                print("🎵 Transcribiendo audio completo")
            
            # Crear directorio para transcripción
            transcription_dir = os.path.join(output_directory, "transcription")
            
            # Ejecutar transcripción
            transcription_results = self.transcription_service.process_audio_transcription(
                audio_path=audio_to_transcribe,
                output_directory=transcription_dir,
                force_language=force_language,
                model_size=transcription_model,
                generate_subtitles=True
            )
            complete_results['transcription'] = transcription_results
            
            # Verificar éxito de transcripción
            if transcription_results['transcription'] and transcription_results['transcription'].success:
                complete_results['success'] = True
                print("\n🎊 ¡PROCESAMIENTO COMPLETO EXITOSO!")
            else:
                print("\n❌ Error en transcripción")
            
        except Exception as e:
            print(f"\n❌ Error inesperado: {str(e)}")
        
        complete_results['total_time'] = time.time() - start_time
        
        # Resumen final
        self._display_complete_summary(complete_results)
        
        return complete_results
    
    def _display_complete_summary(self, results: dict):
        """Muestra resumen del procesamiento completo"""
        print(f"\n📋 RESUMEN FINAL")
        print("=" * 30)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        
        # Estado de cada paso
        download_ok = results['download'] and results['download'].success
        audio_ok = results['audio_processing'] and results['audio_processing']['separation'] and results['audio_processing']['separation'].success
        transcription_ok = results['transcription'] and results['transcription']['transcription'] and results['transcription']['transcription'].success
        
        print(f"📥 Descarga: {'✅' if download_ok else '❌'}")
        print(f"🎵 Audio: {'✅' if audio_ok else '❌'}")
        print(f"🎤 Transcripción: {'✅' if transcription_ok else '❌'}")
        
        if results['success']:
            transcription = results['transcription']['transcription']
            print(f"\n🌍 Idioma detectado: {transcription.language.name}")
            print(f"📝 Palabras: {transcription.word_count}")
            print(f"🔄 Necesita traducción: {'Sí' if transcription.needs_translation else 'No'}")
        
        print(f"\n{'🎉 ¡ÉXITO TOTAL!' if results['success'] else '⚠️  Completado con errores'}")
        
# SERVICIO PRINCIPAL DE TRADUCCIÓN

class TranslationService:
    """Servicio inteligente de traducción con múltiples proveedores"""
    
    def __init__(self, primary_translator: ITranslator, quality_analyzer: Optional[ITranslationQualityAnalyzer] = None):
        """
        Constructor con traductor principal
        
        Args:
            primary_translator: Traductor a usar
            quality_analyzer: Analizador de calidad opcional
        """
        self.primary_translator = primary_translator
        self.quality_analyzer = quality_analyzer or BasicTranslationQualityAnalyzer()
    
    def translate_transcription(self, transcription_text: str, source_language: str, target_language: str = "es", output_directory: Optional[str] = None, progress_callback: Optional[Callable[[TranslationProgress], None]] = None) -> dict:
        """
        Traduce una transcripción completa con análisis de calidad
        
        Args:
            transcription_text: Texto de la transcripción
            source_language: Idioma origen
            target_language: Idioma destino
            output_directory: Directorio para guardar archivos
            progress_callback: Callback para progreso
            
        Returns:
            Dict con resultado completo de traducción
        """
        
        start_time = time.time()
        session_id = f"translation_{int(time.time())}"
        
        results = {
            'translation': None,
            'quality_analysis': None,
            'text_file': None,
            'metadata_file': None,
            'session_info': None,
            'total_time': 0.0
        }
        
        try:
            print(f"🌐 INICIANDO TRADUCCIÓN")
            print("=" * 40)
            print(f"📝 Texto: {len(transcription_text)} caracteres")
            print(f"🔄 {source_language.upper()} → {target_language.upper()}")
            print(f"🤖 Traductor: {self.primary_translator.get_translator_name()}")
            
            # Verificar si realmente necesita traducción
            if source_language.lower() == target_language.lower():
                print(f"🇪🇸 Texto ya está en {target_language.upper()} - no necesita traducción")
                
                # Crear resultado "traducción" que es el mismo texto
                translation_result = TranslationResult(
                    success=True,
                    original_text=transcription_text,
                    translated_text=transcription_text,
                    language_pair=LanguagePair(
                        source_language=source_language,
                        target_language=target_language,
                        source_name=source_language,
                        target_name=target_language
                    ),
                    translator_used="no-translation-needed",
                    processing_time=0.1,
                    confidence_score=1.0,
                    cost_estimate=0.0
                )
                
                results['translation'] = translation_result
            else:
                # Ejecutar traducción real
                print(f"🚀 Ejecutando traducción...")
                
                # Usar traductor de textos largos si está disponible
                if isinstance(self.primary_translator, ILongTextTranslator) and len(transcription_text) > 3000:
                    print(f"📄 Usando traducción para textos largos...")
                    translation_result = self.primary_translator.translate_long_text(
                        text=transcription_text,
                        source_language=source_language,
                        target_language=target_language,
                        max_chunk_size=4000,
                        progress_callback=progress_callback
                    )
                else:
                    # Traducción normal
                    translation_result = self.primary_translator.translate_text(
                        text=transcription_text,
                        source_language=source_language,
                        target_language=target_language
                    )
                
                results['translation'] = translation_result
                
                if not translation_result.success:
                    print(f"❌ Error en traducción: {translation_result.error_message}")
                    results['total_time'] = time.time() - start_time
                    return results
            
            # Análisis de calidad
            if translation_result.success and translation_result.translated_text != transcription_text:
                print(f"📊 Analizando calidad de traducción...")
                
                try:
                    language_pair = translation_result.language_pair or LanguagePair.create_to_spanish(
                        source_language, source_language
                    )
                    
                    quality_metrics = self.quality_analyzer.analyze_translation_quality(
                        original_text=transcription_text,
                        translated_text=translation_result.translated_text,
                        language_pair=language_pair
                    )
                    
                    results['quality_analysis'] = quality_metrics
                    
                    print(f"✅ Calidad: {quality_metrics.get_quality_description()} ({quality_metrics.overall_quality:.1%})")
                    
                except Exception as e:
                    print(f"⚠️  Error en análisis de calidad: {e}")
            
            # Guardar archivos si se especifica directorio
            if output_directory and translation_result.success:
                os.makedirs(output_directory, exist_ok=True)
                
                # Guardar texto traducido
                text_file = os.path.join(output_directory, "translated_text.txt")
                try:
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(translation_result.translated_text)
                    results['text_file'] = text_file
                    print(f"💾 Texto guardado: {os.path.basename(text_file)}")
                except Exception as e:
                    print(f"⚠️  Error guardando texto: {e}")
                
                # Guardar metadata completa
                metadata_file = os.path.join(output_directory, "translation_metadata.json")
                try:
                    metadata = {
                        'session_id': session_id,
                        'translation_result': {
                            'success': translation_result.success,
                            'original_text': translation_result.original_text,
                            'translated_text': translation_result.translated_text,
                            'character_count': translation_result.character_count,
                            'word_count': translation_result.word_count,
                            'translator_used': translation_result.translator_used,
                            'processing_time': translation_result.processing_time,
                            'confidence_score': translation_result.confidence_score,
                            'cost_estimate': translation_result.cost_estimate,
                            'chunks_processed': translation_result.chunks_processed
                        },
                        'language_pair': {
                            'source_language': source_language,
                            'target_language': target_language,
                            'source_name': translation_result.language_pair.source_name if translation_result.language_pair else source_language,
                            'target_name': translation_result.language_pair.target_name if translation_result.language_pair else target_language
                        },
                        'quality_analysis': {
                            'overall_quality': results['quality_analysis'].overall_quality if results['quality_analysis'] else None,
                            'quality_description': results['quality_analysis'].get_quality_description() if results['quality_analysis'] else None,
                            'fluency_score': results['quality_analysis'].fluency_score if results['quality_analysis'] else None,
                            'adequacy_score': results['quality_analysis'].adequacy_score if results['quality_analysis'] else None
                        } if results['quality_analysis'] else None,
                        'timestamp': time.time()
                    }
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    results['metadata_file'] = metadata_file
                    print(f"💾 Metadata guardada: {os.path.basename(metadata_file)}")
                    
                except Exception as e:
                    print(f"⚠️  Error guardando metadata: {e}")
            
            # Crear información de sesión
            processing_end_time = time.time()
            session_info = TranslationSessionInfo(
                session_id=session_id,
                video_title="",  # Se llenará desde contexto superior
                original_language=source_language,
                target_language=target_language,
                total_duration_seconds=0.0,  # Se llenará desde contexto superior
                character_count=translation_result.character_count,
                word_count=translation_result.word_count,
                translator_used=translation_result.translator_used,
                processing_start_time=time.datetime.fromtimestamp(start_time),
                processing_end_time=time.datetime.fromtimestamp(processing_end_time),
                total_cost=translation_result.cost_estimate,
                success=translation_result.success
            )
            
            results['session_info'] = session_info
            results['total_time'] = processing_end_time - start_time
            
            # Resumen final
            self._display_translation_summary(translation_result, results)
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error inesperado en traducción: {e}")
            
            results['total_time'] = processing_time
            results['translation'] = TranslationResult(
                success=False,
                original_text=transcription_text,
                error_message=f"Error inesperado: {str(e)}"
            )
            
            return results
    
    def _display_translation_summary(self, translation: TranslationResult, results: dict):
        """Muestra resumen de la traducción"""
        print(f"\n🎉 TRADUCCIÓN COMPLETADA")
        print("=" * 50)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        print(f"🤖 Traductor usado: {translation.translator_used}")
        print(f"📝 Caracteres: {translation.character_count:,}")
        print(f"📖 Palabras: {translation.word_count:,}")
        print(f"🎯 Confianza: {translation.confidence_score:.1%}")
        
        if translation.cost_estimate > 0:
            print(f"💰 Costo: ${translation.cost_estimate:.4f}")
        else:
            print(f"💰 Costo: Gratis")
        
        if translation.chunks_processed > 1:
            print(f"🔄 Fragmentos procesados: {translation.chunks_processed}")
        
        if results['quality_analysis']:
            quality = results['quality_analysis']
            print(f"📊 Calidad: {quality.get_quality_description()} ({quality.overall_quality:.1%})")
        
        # Archivos generados
        if results['text_file'] or results['metadata_file']:
            print(f"\n📁 Archivos generados:")
            if results['text_file']:
                print(f"   📄 {os.path.basename(results['text_file'])}")
            if results['metadata_file']:
                print(f"   📊 {os.path.basename(results['metadata_file'])}")
        
        # Muestra del texto traducido
        if translation.translated_text:
            sample_length = min(150, len(translation.translated_text))
            sample_text = translation.translated_text[:sample_length]
            if len(translation.translated_text) > sample_length:
                sample_text += "..."
            print(f"\n💬 Muestra de traducción:")
            print(f"   \"{sample_text}\"")

# =============================================================================
# SERVICIO ADAPTATIVO DE TRADUCCIÓN
# =============================================================================

class AdaptiveTranslationService:
    """Servicio que se adapta a los traductores disponibles"""
    
    def __init__(self):
        """Inicializa con detección automática de traductores disponibles"""
        self.available_translators = self._detect_available_translators()
        self.quality_analyzer = BasicTranslationQualityAnalyzer()
        
        if not self.available_translators:
            raise RuntimeError("No hay traductores disponibles")
        
        print(f"🌐 Traductores disponibles: {[t.get_translator_name() for t in self.available_translators]}")
    
    def _detect_available_translators(self) -> List[ITranslator]:
        """Detecta qué traductores están disponibles y configurados"""
        translators = []
        
        # Google Translate (siempre disponible)
        try:
            google = GoogleTranslator()
            if google.is_configured():
                translators.append(google)
                print("✅ Google Translate disponible")
        except Exception as e:
            print(f"❌ Google Translate no disponible: {e}")
        
        # DeepL (si tiene API key)
        try:
            deepl = DeepLTranslator()
            if deepl.is_configured():
                translators.append(deepl)
                print("✅ DeepL disponible")
        except Exception as e:
            print(f"⚠️  DeepL no disponible: {e}")
        
        # OpenAI (si tiene API key y no hay conflictos)
        try:
            openai_translator = OpenAITranslator()
            if openai_translator.is_configured():
                translators.append(openai_translator)
                print("✅ OpenAI Translator disponible")
        except Exception as e:
            print(f"⚠️  OpenAI Translator no disponible: {e}")
        
        return translators
    
    def get_best_translator_for_text(self, text: str, source_language: str) -> ITranslator:
        """Selecciona el mejor traductor para un texto específico"""
        
        text_length = len(text)
        
        # Para textos largos, preferir traductores con chunking
        if text_length > 5000:
            for translator in self.available_translators:
                if isinstance(translator, ILongTextTranslator):
                    if translator.get_translator_name() in ['openai', 'deepl']:
                        return translator
        
        # Para textos medianos, preferir calidad
        if text_length > 1000:
            for translator in self.available_translators:
                if translator.get_translator_name() in ['openai', 'deepl']:
                    return translator
        
        # Para textos cortos, cualquier traductor sirve
        return self.available_translators[0]
    
    def translate_with_best_available(self, text: str, source_language: str, 
                                    target_language: str = "es",
                                    output_directory: Optional[str] = None) -> dict:
        """Traduce usando el mejor traductor disponible"""
        
        best_translator = self.get_best_translator_for_text(text, source_language)
        
        print(f"🎯 Traductor seleccionado: {best_translator.get_translator_name()}")
        
        # Crear servicio de traducción con el traductor seleccionado
        translation_service = TranslationService(
            primary_translator=best_translator,
            quality_analyzer=self.quality_analyzer
        )
        
        return translation_service.translate_transcription(
            transcription_text=text,
            source_language=source_language,
            target_language=target_language,
            output_directory=output_directory
        )
    
    def get_cost_estimates(self, text: str, source_language: str, target_language: str = "es") -> Dict[str, float]:
        """Obtiene estimaciones de costo de todos los traductores disponibles"""
        estimates = {}
        
        for translator in self.available_translators:
            try:
                estimate = translator.estimate_cost(text, source_language, target_language)
                estimates[translator.get_translator_name()] = estimate.estimated_cost_usd
            except Exception as e:
                estimates[translator.get_translator_name()] = -1  # Error
        
        return estimates
    
    def list_available_translators(self) -> List[Dict[str, Any]]:
        """Lista información de traductores disponibles"""
        info = []
        
        for translator in self.available_translators:
            translator_info = {
                'name': translator.get_translator_name(),
                'is_configured': translator.is_configured(),
                'supported_languages': len(translator.get_supported_languages()),
                'supports_long_text': isinstance(translator, ILongTextTranslator)
            }
            info.append(translator_info)
        
        return info

# =============================================================================
# SERVICIO INTEGRADO: TRANSCRIPCIÓN + TRADUCCIÓN
# =============================================================================

class TranscriptionTranslationService:
    """Servicio que integra transcripción y traducción en un solo flujo"""
    
    def __init__(self, transcription_service, translation_service: AdaptiveTranslationService):
        """
        Constructor con servicios de transcripción y traducción
        
        Args:
            transcription_service: Servicio de transcripción (TranscriptionService)
            translation_service: Servicio adaptativo de traducción
        """
        self.transcription_service = transcription_service
        self.translation_service = translation_service
    
    def process_audio_to_spanish(self, audio_path: str, output_directory: str,
                                force_source_language: Optional[str] = None,
                                transcription_model: str = "base") -> dict:
        """
        Procesa audio completo: transcripción → traducción al español
        
        Args:
            audio_path: Ruta del archivo de audio
            output_directory: Directorio de salida
            force_source_language: Forzar idioma origen (opcional)
            transcription_model: Modelo de transcripción a usar
            
        Returns:
            Dict con resultados completos
        """
        
        complete_results = {
            'transcription': None,
            'translation': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            print("🎤→🌐 TRANSCRIPCIÓN + TRADUCCIÓN")
            print("=" * 50)
            
            # Crear directorios
            transcription_dir = os.path.join(output_directory, "transcription")
            translation_dir = os.path.join(output_directory, "translation")
            
            # Paso 1: Transcribir audio
            print("🎤 Paso 1/2: Transcribiendo audio...")
            transcription_results = self.transcription_service.process_audio_transcription(
                audio_path=audio_path,
                output_directory=transcription_dir,
                force_language=force_source_language,
                model_size=transcription_model,
                generate_subtitles=True
            )
            
            complete_results['transcription'] = transcription_results
            
            # Verificar éxito de transcripción
            transcription = transcription_results.get('transcription')
            if not transcription or not transcription.success:
                print("❌ Error en transcripción")
                return complete_results
            
            # Paso 2: Traducir si es necesario
            print("\n🌐 Paso 2/2: Traduciendo al español...")
            
            if transcription.needs_translation:
                translation_results = self.translation_service.translate_with_best_available(
                    text=transcription.text,
                    source_language=transcription.language.code,
                    target_language="es",
                    output_directory=translation_dir
                )
                
                complete_results['translation'] = translation_results
                
                if translation_results['translation'] and translation_results['translation'].success:
                    complete_results['success'] = True
                    print("🎊 ¡Transcripción y traducción completadas!")
                else:
                    print("❌ Error en traducción")
            else:
                print("🇪🇸 Audio ya está en español - no necesita traducción")
                complete_results['success'] = True
                
                # Crear "traducción" que es el mismo texto
                complete_results['translation'] = {
                    'translation': TranslationResult(
                        success=True,
                        original_text=transcription.text,
                        translated_text=transcription.text,
                        translator_used="no-translation-needed",
                        cost_estimate=0.0
                    )
                }
        
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
        
        complete_results['total_time'] = time.time() - start_time
        
        # Resumen final
        self._display_complete_summary(complete_results)
        
        return complete_results
    
    def _display_complete_summary(self, results: dict):
        """Muestra resumen del proceso completo"""
        print(f"\n📋 RESUMEN FINAL")
        print("=" * 30)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        
        # Estados
        transcription_ok = results['transcription'] and results['transcription'].get('transcription') and results['transcription']['transcription'].success
        translation_ok = results['translation'] and results['translation'].get('translation') and results['translation']['translation'].success
        
        print(f"🎤 Transcripción: {'✅' if transcription_ok else '❌'}")
        print(f"🌐 Traducción: {'✅' if translation_ok else '❌'}")
        
        if results['success']:
            transcription = results['transcription']['transcription']
            translation = results['translation']['translation'] if results['translation'] else None
            
            print(f"\n🌍 Idioma original: {transcription.language.name}")
            print(f"📝 Palabras: {transcription.word_count:,}")
            
            if translation and translation.translator_used != "no-translation-needed":
                print(f"🤖 Traductor usado: {translation.translator_used}")
                if translation.cost_estimate > 0:
                    print(f"💰 Costo traducción: ${translation.cost_estimate:.4f}")
                else:
                    print(f"💰 Costo traducción: Gratis")
        
        print(f"\n{'🎉 ¡ÉXITO TOTAL!' if results['success'] else '⚠️  Completado con errores'}")
        
        
# =============================================================================
# SERVICIOS TTS EN ESPAÑOL - AGREGAR A services.py
# =============================================================================



class SpanishTTSService:
    """Servicio principal para generación de audio TTS en español"""
    
    def __init__(self, tts_processor: ISpanishTTSProcessor, voice_manager: ISpanishVoiceManager):
        """
        Constructor con inyección de dependencias
        
        Args:
            tts_processor: Procesador TTS principal
            voice_manager: Gestor de voces españolas
        """
        self.tts_processor = tts_processor
        self.voice_manager = voice_manager
    
    def generate_spanish_audio_from_text(self, text: str, output_directory: str, voice_preference: Optional[str] = None, filename: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> Dict[str, Any]:
        """
        Genera audio en español desde texto con gestión completa
        
        Args:
            text: Texto en español a convertir
            output_directory: Directorio de salida
            voice_preference: Preferencia de voz ("Male", "Female" o ID específico)
            filename: Nombre del archivo (opcional, se genera automáticamente)
            progress_callback: Callback para progreso
            
        Returns:
            Dict con resultado completo del procesamiento
        """
        
        results = {
            'tts_generation': None,
            'audio_file': None,
            'metadata_file': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            # Crear directorio de salida
            os.makedirs(output_directory, exist_ok=True)
            
            # Generar nombre de archivo si no se proporciona
            if not filename:
                # Crear nombre basado en las primeras palabras del texto
                text_preview = text[:30].replace(' ', '_').replace('.', '').replace(',', '')
                filename = f"audio_es_{text_preview}.wav"
            
            # Asegurar extensión .wav
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            output_path = os.path.join(output_directory, filename)
            
            print("🎙️ Generando audio en español...")
            print(f"📝 Caracteres: {len(text)}")
            print(f"🎯 Preferencia de voz: {voice_preference or 'Automática'}")
            
            # Decidir si usar procesamiento largo o normal
            if len(text) > 1000:
                print("📄 Texto largo detectado - usando procesamiento por chunks")
                tts_result = self.tts_processor.process_long_text(
                    text=text,
                    output_path=output_path,
                    max_chunk_size=800,
                    voice_preference=voice_preference,
                    progress_callback=progress_callback
                )
            else:
                print("📝 Procesando texto normal")
                tts_result = self.tts_processor.process_spanish_text(
                    text=text,
                    output_path=output_path,
                    voice_preference=voice_preference,
                    progress_callback=progress_callback
                )
            
            results['tts_generation'] = tts_result
            
            if not tts_result.success:
                print(f"❌ Error en generación TTS: {tts_result.error_message}")
                results['total_time'] = time.time() - start_time
                return results
            
            results['audio_file'] = output_path
            print(f"✅ Audio generado: {filename}")
            print(f"⏱️  Duración: {tts_result.duration_seconds:.1f}s")
            print(f"📊 Tamaño: {tts_result.file_size_mb:.1f}MB")
            if tts_result.voice_used:
                print(f"🎤 Voz: {tts_result.voice_used}")
            
            # Guardar metadata del TTS
            metadata_file = self._save_tts_metadata(output_directory, filename, text, tts_result)
            results['metadata_file'] = metadata_file
            
            results['success'] = True
            print("🎉 Generación TTS completada exitosamente")
            
        except Exception as e:
            print(f"❌ Error inesperado en TTS: {e}")
            results['tts_generation'] = SpanishTTSResult(
                success=False,
                error_message=f"Error inesperado: {str(e)}"
            )
        
        results['total_time'] = time.time() - start_time
        return results
    
    def _save_tts_metadata(self, output_directory: str, filename: str, original_text: str, tts_result: SpanishTTSResult) -> Optional[str]:
        """Guarda metadata del proceso TTS"""
        try:
            metadata = {
                'audio_file': filename,
                'original_text': original_text,
                'character_count': len(original_text),
                'word_count': len(original_text.split()),
                'voice_used': {
                    'id': tts_result.voice_used.id if tts_result.voice_used else None,
                    'name': tts_result.voice_used.name if tts_result.voice_used else None,
                    'locale': tts_result.voice_used.locale if tts_result.voice_used else None,
                    'gender': tts_result.voice_used.gender if tts_result.voice_used else None,
                    'provider': tts_result.voice_used.provider if tts_result.voice_used else None
                },
                'audio_info': {
                    'duration_seconds': tts_result.duration_seconds,
                    'file_size_mb': tts_result.file_size_mb,
                    'processing_time': tts_result.processing_time
                },
                'generation_timestamp': time.time(),
                'success': tts_result.success
            }
            
            metadata_filename = filename.replace('.wav', '_tts_metadata.json')
            metadata_path = os.path.join(output_directory, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return metadata_path
            
        except Exception as e:
            print(f"⚠️  Error guardando metadata TTS: {e}")
            return None
    
    def get_available_voices_info(self) -> Dict[str, Any]:
        """Retorna información detallada de voces disponibles"""
        try:
            all_voices = self.voice_manager.get_available_voices()
            
            voices_by_locale = {}
            voices_by_gender = {'Male': [], 'Female': []}
            
            for voice in all_voices:
                # Agrupar por locale
                if voice.locale not in voices_by_locale:
                    voices_by_locale[voice.locale] = []
                voices_by_locale[voice.locale].append(voice)
                
                # Agrupar por género
                voices_by_gender[voice.gender].append(voice)
            
            return {
                'total_voices': len(all_voices),
                'voices_by_locale': {
                    locale: [{'id': v.id, 'name': v.name, 'gender': v.gender} for v in voices]
                    for locale, voices in voices_by_locale.items()
                },
                'voices_by_gender': {
                    gender: len(voices) for gender, voices in voices_by_gender.items()
                },
                'available_providers': list(set(voice.provider for voice in all_voices)),
                'recommended_female': self.voice_manager.get_recommended_voice("Female"),
                'recommended_male': self.voice_manager.get_recommended_voice("Male")
            }
            
        except Exception as e:
            return {
                'error': f"Error obteniendo información de voces: {e}",
                'total_voices': 0
            }

class TranslationToTTSService:
    """Servicio integrado que convierte traducción → TTS automáticamente"""
    
    def __init__(self, tts_service: SpanishTTSService):
        """
        Constructor con servicio TTS
        
        Args:
            tts_service: Servicio TTS configurado
        """
        self.tts_service = tts_service
    
    def process_translation_to_audio(self, translation_text: str, output_directory: str, voice_preference: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> Dict[str, Any]:
        """
        Procesa traducción directamente a audio
        
        Args:
            translation_text: Texto traducido al español
            output_directory: Directorio de salida
            voice_preference: Preferencia de voz
            progress_callback: Callback para progreso
            
        Returns:
            Dict con resultado completo
        """
        
        print("🔄 CONVERSIÓN TRADUCCIÓN → AUDIO")
        print("=" * 40)
        print(f"📝 Texto traducido: {len(translation_text)} caracteres")
        
        # Generar audio directamente desde la traducción
        result = self.tts_service.generate_spanish_audio_from_text(
            text=translation_text,
            output_directory=output_directory,
            voice_preference=voice_preference,
            filename="translated_audio.wav",
            progress_callback=progress_callback
        )
        
        if result['success']:
            print("🎉 Conversión traducción → audio completada")
            print(f"📁 Audio: {os.path.basename(result['audio_file'])}")
        else:
            print("❌ Error en conversión traducción → audio")
        
        return result

# =============================================================================
# SERVICIO INTEGRADO: TRANSCRIPCIÓN + TRADUCCIÓN + TTS
# =============================================================================

class CompleteVideoToSpanishService:
    """Servicio que integra transcripción + traducción + TTS en español"""
    
    def __init__(self, transcription_translation_service, tts_service: SpanishTTSService):
        """
        Constructor con servicios integrados
        
        Args:
            transcription_translation_service: Servicio transcripción + traducción
            tts_service: Servicio TTS en español
        """
        self.transcription_translation_service = transcription_translation_service
        self.tts_service = tts_service
    
    def process_audio_to_spanish_audio(self, audio_path: str, output_directory: str, voice_preference: Optional[str] = None, transcription_model: str = "base") -> Dict[str, Any]:
        """
        Proceso completo: Audio → Transcripción → Traducción → TTS en español
        
        Args:
            audio_path: Ruta del archivo de audio original
            output_directory: Directorio de salida
            voice_preference: Preferencia de voz para TTS
            transcription_model: Modelo de transcripción a usar
            
        Returns:
            Dict con resultados de todo el pipeline
        """
        
        complete_results = {
            'transcription_translation': None,
            'spanish_tts': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            print("🎤→🌐→🎙️ PIPELINE COMPLETO A ESPAÑOL")
            print("=" * 50)
            
            # Crear subdirectorios
            transcription_dir = os.path.join(output_directory, "transcription_translation")
            tts_dir = os.path.join(output_directory, "spanish_audio")
            
            # Paso 1: Transcripción + Traducción
            print("🎤 Paso 1/2: Transcripción y traducción...")
            transcription_translation_result = self.transcription_translation_service.process_audio_to_spanish(
                audio_path=audio_path,
                output_directory=transcription_dir,
                transcription_model=transcription_model
            )
            
            complete_results['transcription_translation'] = transcription_translation_result
            
            if not transcription_translation_result['success']:
                print("❌ Error en transcripción/traducción")
                return complete_results
            
            # Obtener texto traducido
            translation_result = transcription_translation_result.get('translation', {}).get('translation')
            if not translation_result or not translation_result.translated_text:
                print("❌ No se obtuvo texto traducido")
                return complete_results
            
            spanish_text = translation_result.translated_text
            print(f"✅ Texto en español obtenido: {len(spanish_text)} caracteres")
            
            # Paso 2: Generar audio en español
            print("\n🎙️ Paso 2/2: Generando audio en español...")
            tts_result = self.tts_service.generate_spanish_audio_from_text(
                text=spanish_text,
                output_directory=tts_dir,
                voice_preference=voice_preference,
                filename="final_spanish_audio.wav"
            )
            
            complete_results['spanish_tts'] = tts_result
            
            if tts_result['success']:
                complete_results['success'] = True
                print("\n🎊 ¡PIPELINE COMPLETO EXITOSO!")
                print("🎬 Video procesado completamente al español")
            else:
                print("\n❌ Error en generación de audio español")
            
        except Exception as e:
            print(f"\n❌ Error inesperado en pipeline: {e}")
        
        complete_results['total_time'] = time.time() - start_time
        
        # Resumen final
        self._display_pipeline_summary(complete_results)
        
        return complete_results
    
    def _display_pipeline_summary(self, results: Dict[str, Any]):
        """Muestra resumen del pipeline completo"""
        print(f"\n📋 RESUMEN PIPELINE COMPLETO")
        print("=" * 30)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        
        # Estados de cada paso
        transcription_ok = results['transcription_translation'] and results['transcription_translation']['success']
        tts_ok = results['spanish_tts'] and results['spanish_tts']['success']
        
        print(f"🎤 Transcripción/Traducción: {'✅' if transcription_ok else '❌'}")
        print(f"🎙️ Audio en español: {'✅' if tts_ok else '❌'}")
        
        if results['success']:
            # Información detallada si todo salió bien
            transcription_data = results['transcription_translation']['transcription']['transcription']
            translation_data = results['transcription_translation']['translation']['translation']
            tts_data = results['spanish_tts']['tts_generation']
            
            print(f"\n📊 ESTADÍSTICAS:")
            print(f"🌍 Idioma original: {transcription_data.language.name}")
            print(f"📝 Palabras transcritas: {transcription_data.word_count:,}")
            print(f"🌐 Palabras traducidas: {translation_data.word_count:,}")
            print(f"🎙️ Audio español: {tts_data.duration_seconds:.1f}s")
            if translation_data.cost_estimate > 0:
                print(f"💰 Costo traducción: ${translation_data.cost_estimate:.4f}")
            
            if tts_data.voice_used:
                print(f"🎤 Voz usada: {tts_data.voice_used}")
        
        print(f"\n{'🎉 ¡ÉXITO TOTAL!' if results['success'] else '⚠️  Completado con errores'}")

# Función de conveniencia para uso directo
def create_spanish_audio_from_translation(translation_text: str, output_path: str, 
                                        voice_gender: str = "Female") -> SpanishTTSResult:
    """
    Función de conveniencia para crear audio español desde traducción
    
    Args:
        translation_text: Texto traducido al español
        output_path: Ruta donde guardar el audio
        voice_gender: Género de voz preferido
        
    Returns:
        SpanishTTSResult con el resultado
    """
    from .tts_generators import AdaptiveSpanishTTS
    
    tts = AdaptiveSpanishTTS()
    return tts.generate_with_best_available(translation_text, output_path, voice_gender)        



# =============================================================================
# SERVICIOS DE COMPOSICIÓN DE VIDEO 
# =============================================================================



class VideoCompositionService:
    """Servicio principal para composición de video final"""
    
    def __init__(self, video_composer: IVideoComposer, project_manager: IProjectCompositionManager):
        """
        Constructor con inyección de dependencias
        
        Args:
            video_composer: Compositor de video principal
            project_manager: Gestor de proyectos
        """
        self.video_composer = video_composer
        self.project_manager = project_manager
    
    def compose_video_from_project(self, project_path: str, output_directory: Optional[str] = None, template_name: str = "standard", progress_callback: Optional[Callable[[CompositionProgress], None]] = None) -> Dict[str, Any]:
        """
        Compone video final desde estructura de proyecto
        
        Args:
            project_path: Ruta del proyecto con assets
            output_directory: Directorio de salida (opcional, usa 6_final del proyecto)
            template_name: Template de composición a usar
            progress_callback: Callback para progreso
            
        Returns:
            Dict con resultado completo de composición
        """
        
        results = {
            'composition_result': None,
            'project_validation': None,
            'assets_detected': None,
            'final_video_path': None,
            'metadata_file': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            print("🎬 Iniciando composición de video final")
            print(f"📁 Proyecto: {os.path.basename(project_path)}")
            
            # Validar proyecto
            validation_errors = self.project_manager.validate_project_for_composition(project_path)
            results['project_validation'] = validation_errors
            
            if validation_errors:
                print("❌ Errores de validación encontrados:")
                for error in validation_errors:
                    print(f"   - {error}")
                results['total_time'] = time.time() - start_time
                return results
            
            # Detectar assets automáticamente
            assets = self.project_manager.auto_detect_assets(project_path)
            results['assets_detected'] = assets
            
            print("✅ Assets detectados:")
            print(f"   🎥 Video original: {os.path.basename(assets['original_video']) if assets['original_video'] else 'No encontrado'}")
            print(f"   🎙️ Audio español: {os.path.basename(assets['spanish_audio']) if assets['spanish_audio'] else 'No encontrado'}")
            print(f"   🎵 Música fondo: {os.path.basename(assets['background_music']) if assets['background_music'] else 'No disponible'}")
            
            # Configurar output
            if not output_directory:
                output_directory = os.path.join(project_path, "6_final")
            
            os.makedirs(output_directory, exist_ok=True)
            
            project_name = os.path.basename(project_path)
            output_filename = f"{project_name}_final.mp4"
            output_path = os.path.join(output_directory, output_filename)
            
            # Crear solicitud de composición
            composition_request = VideoCompositionRequest(
                project_name=project_name,
                original_video_path=assets['original_video'],
                spanish_audio_path=assets['spanish_audio'],
                background_music_path=assets['background_music'],
                output_path=output_path,
                spanish_voice_volume=0.8,
                background_music_volume=0.25  # Música más baja para que no compita con voz
            )
            
            print(f"🔄 Componiendo video final...")
            
            # Ejecutar composición
            composition_result = self.video_composer.compose_video(
                composition_request, 
                progress_callback
            )
            
            results['composition_result'] = composition_result
            
            if not composition_result.success:
                print(f"❌ Error en composición: {composition_result.error_message}")
                results['total_time'] = time.time() - start_time
                return results
            
            results['final_video_path'] = composition_result.final_video_path
            print(f"✅ Video final generado: {output_filename}")
            print(f"⏱️  Duración: {composition_result.duration_seconds:.1f}s")
            print(f"📊 Tamaño: {composition_result.file_size_mb:.1f}MB")
            print(f"🎞️ Resolución: {composition_result.video_resolution}")
            print(f"🎵 Pistas audio: {composition_result.audio_tracks_count}")
            
            # Guardar metadata de composición
            metadata_file = self._save_composition_metadata(
                output_directory, project_name, composition_request, composition_result, assets
            )
            results['metadata_file'] = metadata_file
            
            results['success'] = True
            print("🎉 Composición completada exitosamente")
            
        except Exception as e:
            print(f"❌ Error inesperado en composición: {e}")
            results['composition_result'] = VideoCompositionResult(
                success=False,
                error_message=f"Error inesperado: {str(e)}"
            )
        
        results['total_time'] = time.time() - start_time
        return results
    
    def compose_video_from_assets(self, original_video_path: str, spanish_audio_path: str, output_path: str, background_music_path: Optional[str] = None, voice_volume: float = 0.8, music_volume: float = 0.25, progress_callback: Optional[Callable[[CompositionProgress], None]] = None) -> VideoCompositionResult:
        """
        Compone video desde assets específicos
        
        Args:
            original_video_path: Ruta del video original
            spanish_audio_path: Ruta del audio en español
            output_path: Ruta de salida
            background_music_path: Ruta de música de fondo (opcional)
            voice_volume: Volumen de voz (0.0-1.0)
            music_volume: Volumen de música (0.0-1.0)
            progress_callback: Callback para progreso
            
        Returns:
            VideoCompositionResult con el resultado
        """
        
        print("🎬 Composición desde assets específicos")
        print(f"🎥 Video: {os.path.basename(original_video_path)}")
        print(f"🎙️ Audio: {os.path.basename(spanish_audio_path)}")
        if background_music_path:
            print(f"🎵 Música: {os.path.basename(background_music_path)}")
        
        # Crear solicitud
        request = VideoCompositionRequest(
            project_name="custom_composition",
            original_video_path=original_video_path,
            spanish_audio_path=spanish_audio_path,
            background_music_path=background_music_path,
            output_path=output_path,
            spanish_voice_volume=voice_volume,
            background_music_volume=music_volume
        )
        
        # Crear directorio de salida
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ejecutar composición
        return self.video_composer.compose_video(request, progress_callback)
    
    def _save_composition_metadata(self, output_directory: str, project_name: str, request: VideoCompositionRequest, result: VideoCompositionResult, assets: Dict[str, Optional[str]]) -> Optional[str]:
        """Guarda metadata de la composición"""
        try:
            metadata = {
                'project_name': project_name,
                'composition_request': {
                    'original_video': request.original_video_path,
                    'spanish_audio': request.spanish_audio_path,
                    'background_music': request.background_music_path,
                    'spanish_voice_volume': request.spanish_voice_volume,
                    'background_music_volume': request.background_music_volume,
                    'output_format': request.output_format
                },
                'composition_result': {
                    'success': result.success,
                    'final_video_path': result.final_video_path,
                    'duration_seconds': result.duration_seconds,
                    'file_size_mb': result.file_size_mb,
                    'video_resolution': result.video_resolution,
                    'audio_tracks_count': result.audio_tracks_count,
                    'video_codec': result.video_codec,
                    'audio_codec': result.audio_codec,
                    'processing_time': result.processing_time
                },
                'assets_used': assets,
                'composition_timestamp': time.time()
            }
            
            metadata_filename = f"{project_name}_composition_metadata.json"
            metadata_path = os.path.join(output_directory, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return metadata_path
            
        except Exception as e:
            print(f"⚠️  Error guardando metadata de composición: {e}")
            return None
    
    def get_project_composition_info(self, project_path: str) -> Dict[str, Any]:
        """Retorna información de composición disponible para un proyecto"""
        try:
            assets = self.project_manager.auto_detect_assets(project_path)
            validation_errors = self.project_manager.validate_project_for_composition(project_path)
            
            # Analizar assets encontrados
            asset_info = {}
            for asset_type, path in assets.items():
                if path and os.path.exists(path):
                    file_size_mb = os.path.getsize(path) / 1024 / 1024
                    asset_info[asset_type] = {
                        'path': path,
                        'filename': os.path.basename(path),
                        'size_mb': round(file_size_mb, 2),
                        'exists': True
                    }
                else:
                    asset_info[asset_type] = {
                        'exists': False
                    }
            
            return {
                'project_path': project_path,
                'project_name': os.path.basename(project_path),
                'assets': asset_info,
                'validation_errors': validation_errors,
                'ready_for_composition': len(validation_errors) == 0,
                'required_assets': ['original_video', 'spanish_audio'],
                'optional_assets': ['background_music', 'subtitles']
            }
            
        except Exception as e:
            return {
                'error': f"Error analizando proyecto: {e}",
                'ready_for_composition': False
            }

# =============================================================================
# SERVICIO INTEGRADO: PIPELINE COMPLETO CON COMPOSICIÓN
# =============================================================================

class CompleteVideoProcessingService:
    """Servicio que integra TODO el pipeline hasta composición final"""
    
    def __init__(self, complete_video_to_spanish_service, video_composition_service: VideoCompositionService):
        """
        Constructor con servicios integrados
        
        Args:
            complete_video_to_spanish_service: Servicio transcripción + traducción + TTS
            video_composition_service: Servicio de composición de video
        """
        self.complete_video_to_spanish_service = complete_video_to_spanish_service
        self.video_composition_service = video_composition_service
    
    def process_video_url_to_final_spanish_video(self, video_url: str, project_name: Optional[str] = None,
                                                voice_preference: str = "Female") -> Dict[str, Any]:
        """
        Pipeline completo: URL → Video final en español
        
        Args:
            video_url: URL del video a procesar
            project_name: Nombre del proyecto (opcional)
            voice_preference: Preferencia de voz TTS
            
        Returns:
            Dict con resultado completo del pipeline
        """
        
        pipeline_results = {
            'download_and_processing': None,
            'video_composition': None,
            'final_video_path': None,
            'total_time': 0.0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            print("🚀 PIPELINE COMPLETO: URL → VIDEO FINAL EN ESPAÑOL")
            print("=" * 60)
            print(f"🔗 URL: {video_url}")
            
            # Paso 1: Procesar video (descarga → audio → transcripción → traducción → TTS)
            print("\n📥 Fase 1: Procesamiento completo (descarga + audio + traducción + TTS)")
            
            # Necesitamos obtener la ruta del proyecto del servicio de descarga
            # Este paso incluye todo hasta generar el audio en español
            processing_result = self.complete_video_to_spanish_service.process_audio_to_spanish_audio(
                audio_path=None,  # Se obtendrá del video descargado
                output_directory="temp_processing",  # Directorio temporal
                voice_preference=voice_preference
            )
            
            pipeline_results['download_and_processing'] = processing_result
            
            if not processing_result['success']:
                print("❌ Error en procesamiento inicial")
                return pipeline_results
            
            print("✅ Procesamiento inicial completado")
            
            # Paso 2: Composición final del video
            print("\n🎬 Fase 2: Composición de video final")
            
            # Aquí necesitaríamos la ruta del proyecto generado
            # Por ahora simulamos con el directorio temporal
            project_path = "temp_processing"  # Esto vendría del resultado anterior
            
            composition_result = self.video_composition_service.compose_video_from_project(
                project_path=project_path
            )
            
            pipeline_results['video_composition'] = composition_result
            
            if composition_result['success']:
                pipeline_results['final_video_path'] = composition_result['final_video_path']
                pipeline_results['success'] = True
                
                print("🎊 ¡PIPELINE COMPLETO EXITOSO!")
                print(f"🎬 Video final: {os.path.basename(composition_result['final_video_path'])}")
            else:
                print("❌ Error en composición final")
            
        except Exception as e:
            print(f"❌ Error inesperado en pipeline: {e}")
        
        pipeline_results['total_time'] = time.time() - start_time
        
        # Resumen final
        self._display_pipeline_summary(pipeline_results)
        
        return pipeline_results
    
    def _display_pipeline_summary(self, results: Dict[str, Any]):
        """Muestra resumen del pipeline completo"""
        print(f"\n📋 RESUMEN PIPELINE COMPLETO")
        print("=" * 30)
        print(f"⏱️  Tiempo total: {results['total_time']:.1f}s")
        
        # Estados
        processing_ok = results['download_and_processing'] and results['download_and_processing']['success']
        composition_ok = results['video_composition'] and results['video_composition']['success']
        
        print(f"📥 Procesamiento inicial: {'✅' if processing_ok else '❌'}")
        print(f"🎬 Composición final: {'✅' if composition_ok else '❌'}")
        
        if results['success']:
            print(f"\n🎬 Video final listo: {results['final_video_path']}")
            print("🎉 ¡Video completamente traducido al español!")
        
        print(f"\n{'🎊 ¡ÉXITO TOTAL!' if results['success'] else '⚠️  Completado con errores'}")

# Función de conveniencia para uso directo
def compose_project_video(project_path: str, template: str = "standard") -> VideoCompositionResult:
    """
    Función de conveniencia para componer video desde proyecto
    
    Args:
        project_path: Ruta del proyecto
        template: Template de composición
        
    Returns:
        VideoCompositionResult con el resultado
    """
    from .video_composers import AdaptiveVideoComposer, ProjectCompositionManager
    
    composer = AdaptiveVideoComposer()
    project_manager = ProjectCompositionManager()
    
    service = VideoCompositionService(composer, project_manager)
    
    result = service.compose_video_from_project(project_path)
    return result['composition_result'] if result['composition_result'] else VideoCompositionResult(
        success=False,
        error_message="Error en servicio de composición"
    )