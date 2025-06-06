"""Implementaciones de transcripción multiidioma con faster-whisper"""

import os
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator

from .interfaces import ITranscriber, ISubtitleGenerator, ILanguageDetector
from .models import (TranscriptionResult, TranscriptionSegment, LanguageInfo, SubtitleGenerationResult)

class WhisperMultilingualTranscriber(ITranscriber, ILanguageDetector):
    """Transcriptor multiidioma usando faster-whisper"""
    
    def __init__(self):
        self._model = None
        self._model_size = None
        self._check_whisper_availability()
    
    def _check_whisper_availability(self):
        """Verifica que faster-whisper esté disponible"""
        try:
            from faster_whisper import WhisperModel
            print("✅ faster-whisper multiidioma disponible")
        except ImportError:
            raise ImportError(
                "faster-whisper no está instalado. Instalar con: pip install faster-whisper"
            )
    
    def _load_model(self, model_size: str = "base"):
        """Carga el modelo faster-whisper de forma lazy"""
        if self._model is None or self._model_size != model_size:
            from faster_whisper import WhisperModel
            
            print(f"🤖 Cargando modelo faster-whisper '{model_size}' multiidioma...")
            
            # Información sobre modelos
            model_info = {
                "tiny": "39 MB - Muy rápido, calidad básica",
                "base": "74 MB - Rápido, buena calidad", 
                "small": "244 MB - Lento, muy buena calidad",
                "medium": "769 MB - Muy lento, excelente calidad",
                "large-v1": "1550 MB - Muy lento, calidad máxima",
                "large-v2": "1550 MB - Muy lento, mejor calidad",
                "large-v3": "1550 MB - Muy lento, última versión"
            }
            
            print(f"📊 Modelo '{model_size}': {model_info.get(model_size, 'Tamaño desconocido')}")
            
            # Crear modelo con faster-whisper
            self._model = WhisperModel(
                model_size, 
                device="auto",  # Detecta automáticamente CPU/GPU
                compute_type="float32"  # Compatibilidad máxima
            )
            self._model_size = model_size
            
            print(f"✅ Modelo faster-whisper '{model_size}' cargado correctamente")
    
    def get_supported_languages(self) -> List[str]:
        """Retorna lista de idiomas soportados por Whisper"""
        # faster-whisper soporta los mismos 99 idiomas que whisper original
        return [
            'es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi',
            'th', 'vi', 'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'he', 'cs', 'sk',
            'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'uk', 'be', 'mk',
            'sq', 'eu', 'gl', 'ca', 'cy', 'ga', 'mt', 'is', 'fo', 'gd', 'kw', 'br',
            'co', 'oc', 'ms', 'id', 'tl', 'sw', 'yo', 'ig', 'ha', 'zu', 'af', 'xh',
            'st', 'tn', 've', 'ss', 'ts', 'nr', 'nso', 'mg', 'am', 'ti', 'om', 'so',
            'rw', 'rn', 'ny', 'kg', 'ln', 'lg', 'ak', 'tw', 'ff', 'wo', 'bm', 'sn',
            'ee', 'kr', 'ca', 'ceb', 'ny', 'hy', 'as', 'ay', 'az', 'ba', 'bn', 'bho', 
            'bs', 'mi', 'jw', 'su', 'yue'
        ]
    
    def detect_language(self, audio_path: str) -> LanguageInfo:
        """Detecta idioma usando faster-whisper"""
        try:
            self._load_model("tiny")  # Modelo pequeño para detección rápida
            
            print("🔍 Detectando idioma del audio...")
            
            # Con faster-whisper, detectamos idioma transcribiendo una muestra pequeña
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=1,  # Rápido
                language=None,  # Detección automática
                condition_on_previous_text=False,
                initial_prompt=None,
                vad_filter=True,  # Filtro de actividad de voz
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            detected_language = info.language
            confidence = info.language_probability
            
            language_info = LanguageInfo.from_whisper_code(detected_language, confidence)
            
            print(f"🌍 Idioma detectado: {language_info.name} ({language_info.code})")
            print(f"📊 Confianza: {confidence:.1%}")
            
            return language_info
            
        except Exception as e:
            print(f"⚠️  Error detectando idioma: {e}")
            # Fallback a inglés
            return LanguageInfo.from_whisper_code("en", 0.5)
    
    def transcribe_audio(self, audio_path: str, 
                        force_language: Optional[str] = None,
                        model_size: str = "base") -> TranscriptionResult:
        """Transcribe audio con detección automática de idioma usando faster-whisper"""
        
        start_time = time.time()
        
        try:
            # Verificar archivo
            if not os.path.exists(audio_path):
                return TranscriptionResult(
                    success=False,
                    error_message=f"Archivo de audio no encontrado: {audio_path}"
                )
            
            # Cargar modelo
            self._load_model(model_size)
            
            print(f"🎤 Iniciando transcripción multiidioma...")
            print(f"📁 Archivo: {os.path.basename(audio_path)}")
            print(f"🤖 Modelo: faster-whisper-{model_size}")
            
            # Configurar idioma
            language = force_language if force_language else None
            if force_language:
                print(f"🔒 Idioma forzado: {force_language}")
            else:
                print("🔍 Detección automática de idioma habilitada")
            
            # Ejecutar transcripción con faster-whisper
            print("🧠 Ejecutando transcripción con IA...")
            
            segments_iterator, info = self._model.transcribe(
                audio_path,
                beam_size=5,  # Balance entre velocidad y calidad
                language=language,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=True,  # Timestamps por palabra
                vad_filter=True,  # Filtro de actividad de voz
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    max_speech_duration_s=float('inf')
                )
            )
            
            # Convertir iterator a lista y procesar segmentos
            print("📝 Procesando segmentos...")
            segments = []
            full_text_parts = []
            
            for segment in segments_iterator:
                # Crear segmento
                segment_obj = TranscriptionSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0
                )
                segments.append(segment_obj)
                full_text_parts.append(segment.text.strip())
            
            # Texto completo
            full_text = ' '.join(full_text_parts).strip()
            
            # Información del idioma detectado
            detected_language = info.language
            language_probability = info.language_probability
            
            processing_time = time.time() - start_time
            
            # Crear información de idioma
            language_info = LanguageInfo.from_whisper_code(detected_language, language_probability)
            
            # Calcular duración del audio (from info or estimate)
            audio_duration = info.duration if hasattr(info, 'duration') else (segments[-1].end_time if segments else 0.0)
            
            print(f"✅ Transcripción completada en {processing_time:.1f}s")
            print(f"🌍 Idioma detectado: {language_info.name} ({language_info.code})")
            print(f"📊 Probabilidad idioma: {language_probability:.1%}")
            print(f"📝 Texto transcrito: {len(full_text)} caracteres")
            print(f"🎬 Segmentos: {len(segments)}")
            print(f"⏱️  Duración audio: {audio_duration:.1f}s")
            
            # Mostrar muestra del texto
            if full_text:
                sample_text = full_text[:100] + "..." if len(full_text) > 100 else full_text
                print(f"💬 Muestra: {sample_text}")
            
            # Determinar si necesita traducción
            needs_translation = not language_info.is_spanish
            if needs_translation:
                print(f"🔄 Traducción necesaria: {language_info.name} → Español")
            else:
                print(f"🇪🇸 Ya está en español - no necesita traducción")
            
            return TranscriptionResult(
                success=True,
                text=full_text,
                language=language_info,
                segments=segments,
                model_used=f"faster-whisper-{model_size}",
                processing_time=processing_time,
                audio_duration=audio_duration,
                needs_translation=needs_translation
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error en transcripción: {str(e)}")
            return TranscriptionResult(
                success=False,
                error_message=f"Error en transcripción: {str(e)}",
                processing_time=processing_time
            )

class SRTSubtitleGenerator(ISubtitleGenerator):
    """Generador de subtítulos SRT y VTT"""
    
    def generate_srt(self, transcription: TranscriptionResult, 
                    output_path: str) -> SubtitleGenerationResult:
        """Genera archivo SRT desde transcripción"""
        
        try:
            if not transcription.segments:
                return SubtitleGenerationResult(
                    success=False,
                    error_message="No hay segmentos para generar subtítulos"
                )
            
            print(f"📄 Generando subtítulos SRT...")
            print(f"🎬 Segmentos: {len(transcription.segments)}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generar contenido SRT
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(transcription.segments, 1):
                    srt_content = segment.to_srt_format(i)
                    f.write(srt_content + "\n")
            
            total_duration = transcription.audio_duration
            
            print(f"✅ SRT generado: {os.path.basename(output_path)}")
            print(f"⏱️  Duración total: {total_duration:.1f}s")
            
            return SubtitleGenerationResult(
                success=True,
                srt_file_path=output_path,
                segment_count=len(transcription.segments),
                total_duration=total_duration
            )
            
        except Exception as e:
            return SubtitleGenerationResult(
                success=False,
                error_message=f"Error generando SRT: {str(e)}"
            )
    
    def generate_vtt(self, transcription: TranscriptionResult, 
                    output_path: str) -> SubtitleGenerationResult:
        """Genera archivo VTT desde transcripción"""
        
        try:
            if not transcription.segments:
                return SubtitleGenerationResult(
                    success=False,
                    error_message="No hay segmentos para generar subtítulos VTT"
                )
            
            print(f"📄 Generando subtítulos VTT...")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generar contenido VTT
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for segment in transcription.segments:
                    start_time = self._format_vtt_time(segment.start_time)
                    end_time = self._format_vtt_time(segment.end_time)
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")
            
            print(f"✅ VTT generado: {os.path.basename(output_path)}")
            
            return SubtitleGenerationResult(
                success=True,
                srt_file_path=output_path,
                segment_count=len(transcription.segments),
                total_duration=transcription.audio_duration
            )
            
        except Exception as e:
            return SubtitleGenerationResult(
                success=False,
                error_message=f"Error generando VTT: {str(e)}"
            )
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Convierte segundos a formato VTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

class FallbackTranscriber(ITranscriber):
    """Transcriptor de fallback usando speech_recognition (opcional)"""
    
    def __init__(self):
        self._check_fallback_availability()
    
    def _check_fallback_availability(self):
        """Verifica disponibilidad de transcriptor fallback"""
        try:
            import speech_recognition as sr
            print("✅ SpeechRecognition disponible como fallback")
        except ImportError:
            print("⚠️  SpeechRecognition no disponible - solo faster-whisper")
    
    def get_supported_languages(self) -> List[str]:
        """Idiomas soportados por Google Speech Recognition"""
        return ['es', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
    
    def transcribe_audio(self, audio_path: str, 
                        force_language: Optional[str] = None,
                        model_size: str = "base") -> TranscriptionResult:
        """Transcripción básica usando Google Speech Recognition"""
        
        start_time = time.time()
        
        try:
            import speech_recognition as sr
            
            print("🔄 Usando transcriptor de fallback (Google Speech Recognition)...")
            
            # Crear recognizer
            r = sr.Recognizer()
            
            # Cargar audio
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
            
            # Configurar idioma
            language_code = force_language if force_language else "es-ES"
            
            # Transcribir
            text = r.recognize_google(audio, language=language_code)
            
            processing_time = time.time() - start_time
            
            # Crear resultado básico
            language_info = LanguageInfo.from_whisper_code(
                force_language if force_language else "es", 0.7
            )
            
            return TranscriptionResult(
                success=True,
                text=text,
                language=language_info,
                segments=[],  # Google SR no da segmentos
                model_used="google-speech-recognition",
                processing_time=processing_time,
                audio_duration=0.0,
                needs_translation=(force_language != "es")
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TranscriptionResult(
                success=False,
                error_message=f"Error en transcripción fallback: {str(e)}",
                processing_time=processing_time
            )