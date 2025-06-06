# -*- coding: utf-8 -*-
"""
Implementaciones de Text-to-Speech enfocadas en español
"""

import os
import time
import hashlib
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from .interfaces import (
    ISpanishTTSProvider, ISpanishVoiceManager, ISpanishTTSProcessor, ISpanishTTSQualityAnalyzer, ISpanishTextPreprocessor, IAdaptiveSpanishTTS)
from .models import (SpanishVoice, SpanishTTSRequest, SpanishTTSResult, TTSProgress, SpanishVoiceFilter)

class EdgeTTSSpanishProvider(ISpanishTTSProvider):
    """Proveedor TTS usando Edge (Microsoft) - Gratis y de alta calidad"""
    
    def __init__(self):
        self._check_availability()
        self._voices_cache = None
    
    def _check_availability(self):
        """Verifica que Edge TTS esté disponible"""
        try:
            import edge_tts
            self._edge_available = True
        except ImportError:
            self._edge_available = False
            print("⚠️  Edge TTS no disponible. Instalar: pip install edge-tts")
    
    def is_available(self) -> bool:
        return self._edge_available
    
    def get_provider_name(self) -> str:
        return "edge"
    
    def get_supported_voices(self) -> List[SpanishVoice]:
        """Retorna voces españolas disponibles en Edge TTS"""
        if not self._edge_available:
            return []
        
        if self._voices_cache is not None:
            return self._voices_cache
        
        try:
            import edge_tts
            
            # Voces españolas conocidas de Edge TTS
            spanish_voices = [
                # España
                SpanishVoice(
                    id="es-ES-AlvaroNeural",
                    name="Álvaro",
                    locale="es-ES",
                    gender="Male",
                    provider="edge",
                    is_neural=True
                ),
                SpanishVoice(
                    id="es-ES-ElviraNeural", 
                    name="Elvira",
                    locale="es-ES",
                    gender="Female",
                    provider="edge",
                    is_neural=True
                ),
                # México
                SpanishVoice(
                    id="es-MX-DaliaNeural",
                    name="Dalia",
                    locale="es-MX",
                    gender="Female", 
                    provider="edge",
                    is_neural=True
                ),
                SpanishVoice(
                    id="es-MX-JorgeNeural",
                    name="Jorge",
                    locale="es-MX",
                    gender="Male",
                    provider="edge",
                    is_neural=True
                ),
                # Argentina
                SpanishVoice(
                    id="es-AR-ElenaNeural",
                    name="Elena",
                    locale="es-AR",
                    gender="Female",
                    provider="edge",
                    is_neural=True
                ),
                SpanishVoice(
                    id="es-AR-TomasNeural",
                    name="Tomás",
                    locale="es-AR",
                    gender="Male",
                    provider="edge",
                    is_neural=True
                ),
                # Colombia
                SpanishVoice(
                    id="es-CO-SalomeNeural",
                    name="Salomé",
                    locale="es-CO",
                    gender="Female",
                    provider="edge",
                    is_neural=True
                ),
                SpanishVoice(
                    id="es-CO-GonzaloNeural",
                    name="Gonzalo",
                    locale="es-CO",
                    gender="Male",
                    provider="edge",
                    is_neural=True
                )
            ]
            
            self._voices_cache = spanish_voices
            return spanish_voices
            
        except Exception as e:
            print(f"Error obteniendo voces Edge: {e}")
            return []
    
    def validate_text(self, text: str) -> bool:
        """Valida texto para Edge TTS"""
        if not text or not text.strip():
            return False
        if len(text) > 10000:  # Límite de Edge TTS
            return False
        return True
    
    async def _generate_speech_async(self, request: SpanishTTSRequest, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """Generación asíncrona con Edge TTS"""
        try:
            import edge_tts
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="preparing",
                    percentage=10,
                    current_chunk=1,
                    total_chunks=1,
                    current_text_preview=request.text[:50] + "..."
                ))
            
            # Configurar comunicación con Edge TTS
            communicate = edge_tts.Communicate(
                text=request.text,
                voice=request.voice.id,
                rate=f"{int((request.speed - 1) * 50):+d}%",  # Convertir a formato Edge
                volume=f"{int(request.volume * 100):+d}%"
            )
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="generating",
                    percentage=50,
                    current_chunk=1,
                    total_chunks=1
                ))
            
            # Generar y guardar audio
            await communicate.save(request.output_path)
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="completed",
                    percentage=100,
                    current_chunk=1,
                    total_chunks=1
                ))
            
            # Calcular estadísticas del archivo generado
            file_stats = Path(request.output_path).stat()
            file_size_mb = file_stats.st_size / 1024 / 1024
            
            # Estimar duración (aproximación: 150 palabras por minuto)
            word_count = len(request.text.split())
            estimated_duration = (word_count / 150) * 60 / request.speed
            
            return SpanishTTSResult(
                success=True,
                audio_file_path=request.output_path,
                duration_seconds=estimated_duration,
                file_size_mb=file_size_mb,
                voice_used=request.voice,
                character_count=len(request.text)
            )
            
        except Exception as e:
            return SpanishTTSResult(
                success=False,
                error_message=f"Error Edge TTS: {str(e)}"
            )
    
    def generate_speech(self, request: SpanishTTSRequest, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """Genera audio usando Edge TTS (wrapper síncrono)"""
        if not self.is_available():
            return SpanishTTSResult(
                success=False,
                error_message="Edge TTS no está disponible"
            )
        
        if not self.validate_text(request.text):
            return SpanishTTSResult(
                success=False,
                error_message="Texto no válido para TTS"
            )
        
        start_time = time.time()
        
        try:
            # Ejecutar generación asíncrona
            result = asyncio.run(self._generate_speech_async(request, progress_callback))
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            return SpanishTTSResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=f"Error en generación: {str(e)}"
            )

class SpanishVoiceManager(ISpanishVoiceManager):
    """Gestor de voces españolas disponibles"""
    
    def __init__(self):
        self.providers = [
            EdgeTTSSpanishProvider()
        ]
        self._all_voices = None
    
    def get_available_voices(self) -> List[SpanishVoice]:
        """Retorna todas las voces españolas disponibles"""
        if self._all_voices is not None:
            return self._all_voices
        
        all_voices = []
        for provider in self.providers:
            if provider.is_available():
                voices = provider.get_supported_voices()
                all_voices.extend(voices)
        
        self._all_voices = all_voices
        return all_voices
    
    def filter_voices(self, filter_criteria: SpanishVoiceFilter) -> List[SpanishVoice]:
        """Filtra voces según criterios"""
        available_voices = self.get_available_voices()
        return [voice for voice in available_voices if filter_criteria.matches(voice)]
    
    def get_recommended_voice(self, gender: str = "Female") -> Optional[SpanishVoice]:
        """Retorna voz recomendada"""
        # Orden de preferencia: es-ES > es-MX > otros
        preferred_locales = ["es-ES", "es-MX", "es-AR", "es-CO"]
        
        for locale in preferred_locales:
            filter_criteria = SpanishVoiceFilter(
                locale=locale,
                gender=gender,
                neural_only=True
            )
            matching_voices = self.filter_voices(filter_criteria)
            if matching_voices:
                return matching_voices[0]
        
        # Fallback: cualquier voz del género solicitado
        filter_criteria = SpanishVoiceFilter(gender=gender, neural_only=True)
        matching_voices = self.filter_voices(filter_criteria)
        return matching_voices[0] if matching_voices else None
    
    def get_voice_by_id(self, voice_id: str) -> Optional[SpanishVoice]:
        """Busca voz específica por ID"""
        available_voices = self.get_available_voices()
        for voice in available_voices:
            if voice.id == voice_id:
                return voice
        return None

class SpanishTextPreprocessor(ISpanishTextPreprocessor):
    """Preprocesador de texto en español para TTS"""
    
    def clean_text_for_tts(self, text: str) -> str:
        """Limpia texto para optimizar TTS"""
        # Remover caracteres problemáticos
        text = re.sub(r'[^\w\s.,;:!?¡¿\-\'\"]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Asegurar puntuación al final
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def normalize_spanish_text(self, text: str) -> str:
        """Normaliza texto en español"""
        # Expandir números comunes
        number_replacements = {
            r'\b1\b': 'uno',
            r'\b2\b': 'dos', 
            r'\b3\b': 'tres',
            r'\b4\b': 'cuatro',
            r'\b5\b': 'cinco',
            r'\b6\b': 'seis',
            r'\b7\b': 'siete',
            r'\b8\b': 'ocho',
            r'\b9\b': 'nueve',
            r'\b10\b': 'diez'
        }
        
        for pattern, replacement in number_replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Expandir abreviaciones comunes
        abbreviations = {
            r'\bDr\.\b': 'Doctor',
            r'\bDra\.\b': 'Doctora',
            r'\bSr\.\b': 'Señor',
            r'\bSra\.\b': 'Señora',
            r'\bEtc\.\b': 'etcétera',
            r'\betc\.\b': 'etcétera'
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones apropiadas para TTS"""
        # Divisor básico por puntuación
        sentences = re.split(r'[.!?]+', text)
        
        # Limpiar y filtrar oraciones vacías
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Asegurar que termine con puntuación
                if not sentence[-1] in '.!?':
                    sentence += '.'
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def estimate_audio_duration(self, text: str, voice: SpanishVoice) -> float:
        """Estima duración del audio"""
        word_count = len(text.split())
        # Aproximación: 150 palabras por minuto en español
        duration_minutes = word_count / 150
        return duration_minutes * 60

class SpanishTTSProcessor(ISpanishTTSProcessor):
    """Procesador principal de TTS en español"""
    
    def __init__(self):
        self.voice_manager = SpanishVoiceManager()
        self.text_preprocessor = SpanishTextPreprocessor()
        self.providers = {
            "edge": EdgeTTSSpanishProvider()
        }
    
    def get_available_providers(self) -> List[str]:
        """Retorna proveedores disponibles"""
        available = []
        for name, provider in self.providers.items():
            if provider.is_available():
                available.append(name)
        return available
    
    def _get_voice_for_preference(self, voice_preference: Optional[str]) -> Optional[SpanishVoice]:
        """Obtiene voz basada en preferencia"""
        if not voice_preference:
            return self.voice_manager.get_recommended_voice("Female")
        
        # Intentar buscar por ID específico
        voice = self.voice_manager.get_voice_by_id(voice_preference)
        if voice:
            return voice
        
        # Intentar buscar por género
        if voice_preference.lower() in ["male", "female"]:
            return self.voice_manager.get_recommended_voice(voice_preference.title())
        
        # Fallback
        return self.voice_manager.get_recommended_voice("Female")
    
    def process_spanish_text(self, text: str, output_path: str, voice_preference: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """Procesa texto completo en español"""
        
        start_time = time.time()
        
        try:
            # Preprocesar texto
            clean_text = self.text_preprocessor.clean_text_for_tts(text)
            normalized_text = self.text_preprocessor.normalize_spanish_text(clean_text)
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="preparing",
                    percentage=20,
                    current_chunk=1,
                    total_chunks=1,
                    current_text_preview=normalized_text[:50] + "..."
                ))
            
            # Seleccionar voz
            voice = self._get_voice_for_preference(voice_preference)
            if not voice:
                return SpanishTTSResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No se encontró voz española disponible"
                )
            
            # Crear solicitud TTS
            request = SpanishTTSRequest(
                text=normalized_text,
                voice=voice,
                output_path=output_path
            )
            
            # Obtener proveedor apropiado
            provider = self.providers.get(voice.provider)
            if not provider or not provider.is_available():
                return SpanishTTSResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message=f"Proveedor {voice.provider} no disponible"
                )
            
            # Generar audio
            result = provider.generate_speech(request, progress_callback)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return SpanishTTSResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=f"Error procesando texto: {str(e)}"
            )
    
    def process_long_text(self, text: str, output_path: str, max_chunk_size: int = 1000, voice_preference: Optional[str] = None, progress_callback: Optional[Callable[[TTSProgress], None]] = None) -> SpanishTTSResult:
        """Procesa textos largos dividiéndolos en chunks"""
        
        start_time = time.time()
        
        try:
            # Dividir texto en oraciones
            sentences = self.text_preprocessor.split_into_sentences(text)
            
            # Agrupar oraciones en chunks
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) <= max_chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if not chunks:
                return SpanishTTSResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No se pudieron crear chunks del texto"
                )
            
            # Seleccionar voz
            voice = self._get_voice_for_preference(voice_preference)
            if not voice:
                return SpanishTTSResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No se encontró voz española disponible"
                )
            
            # Procesar cada chunk
            audio_files = []
            total_duration = 0.0
            
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    percentage = (i / len(chunks)) * 80 + 10  # 10-90%
                    progress_callback(TTSProgress(
                        stage="generating",
                        percentage=percentage,
                        current_chunk=i + 1,
                        total_chunks=len(chunks),
                        current_text_preview=chunk[:50] + "..."
                    ))
                
                # Archivo temporal para este chunk
                chunk_path = output_path.replace('.wav', f'_chunk_{i}.wav')
                
                # Procesar chunk individual
                chunk_result = self.process_spanish_text(
                    text=chunk,
                    output_path=chunk_path,
                    voice_preference=voice_preference
                )
                
                if not chunk_result.success:
                    # Limpiar archivos temporales
                    for temp_file in audio_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    
                    return SpanishTTSResult(
                        success=False,
                        processing_time=time.time() - start_time,
                        error_message=f"Error en chunk {i+1}: {chunk_result.error_message}"
                    )
                
                audio_files.append(chunk_path)
                total_duration += chunk_result.duration_seconds
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="saving",
                    percentage=95,
                    current_chunk=len(chunks),
                    total_chunks=len(chunks)
                ))
            
            # Combinar archivos de audio
            success = self._combine_audio_files(audio_files, output_path)
            
            # Limpiar archivos temporales
            for temp_file in audio_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            if not success:
                return SpanishTTSResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="Error combinando archivos de audio"
                )
            
            # Calcular estadísticas finales
            file_stats = Path(output_path).stat()
            file_size_mb = file_stats.st_size / 1024 / 1024
            
            if progress_callback:
                progress_callback(TTSProgress(
                    stage="completed",
                    percentage=100,
                    current_chunk=len(chunks),
                    total_chunks=len(chunks)
                ))
            
            return SpanishTTSResult(
                success=True,
                audio_file_path=output_path,
                duration_seconds=total_duration,
                file_size_mb=file_size_mb,
                voice_used=voice,
                processing_time=time.time() - start_time,
                character_count=len(text)
            )
            
        except Exception as e:
            return SpanishTTSResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=f"Error procesando texto largo: {str(e)}"
            )
    
    def _combine_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """Combina múltiples archivos de audio en uno solo"""
        try:
            import subprocess
            
            # Crear lista de archivos para ffmpeg
            concat_list = output_path.replace('.wav', '_concat_list.txt')
            
            with open(concat_list, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            # Comando ffmpeg para concatenar
            command = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list, '-c', 'copy', output_path
            ]
            
            result = subprocess.run(command, capture_output=True, check=True)
            
            # Limpiar archivo temporal
            try:
                os.remove(concat_list)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"Error combinando audio: {e}")
            return False

class AdaptiveSpanishTTS(IAdaptiveSpanishTTS):
    """TTS adaptativo que maneja múltiples proveedores automáticamente"""
    
    def __init__(self):
        self.processor = SpanishTTSProcessor()
        self.provider_rankings = ["edge"]  # Orden de preferencia
    
    def get_provider_rankings(self) -> List[str]:
        """Retorna proveedores ordenados por preferencia"""
        available_providers = self.processor.get_available_providers()
        return [provider for provider in self.provider_rankings if provider in available_providers]
    
    def fallback_to_next_provider(self, failed_provider: str) -> Optional[ISpanishTTSProvider]:
        """Cambia al siguiente proveedor disponible"""
        rankings = self.get_provider_rankings()
        
        try:
            current_index = rankings.index(failed_provider)
            if current_index + 1 < len(rankings):
                next_provider_name = rankings[current_index + 1]
                return self.processor.providers.get(next_provider_name)
        except ValueError:
            pass
        
        return None
    
    def generate_with_best_available(self, text: str, output_path: str, voice_preference: Optional[str] = None) -> SpanishTTSResult:
        """Genera audio usando el mejor proveedor disponible"""
        
        available_providers = self.get_provider_rankings()
        
        if not available_providers:
            return SpanishTTSResult(
                success=False,
                error_message="No hay proveedores TTS disponibles"
            )
        
        last_error = None
        
        # Intentar con cada proveedor en orden de preferencia
        for provider_name in available_providers:
            try:
                result = self.processor.process_spanish_text(
                    text=text,
                    output_path=output_path,
                    voice_preference=voice_preference
                )
                
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    continue
                    
            except Exception as e:
                last_error = str(e)
                continue
        
        return SpanishTTSResult(
            success=False,
            error_message=f"Todos los proveedores fallaron. Último error: {last_error}"
        )

# Funciones de conveniencia para uso fácil
def generate_spanish_audio(text: str, output_path: str, voice_gender: str = "Female") -> SpanishTTSResult:
    """
    Función de conveniencia para generar audio en español
    
    Args:
        text: Texto en español a convertir
        output_path: Ruta donde guardar el audio
        voice_gender: Género de voz preferido ("Male" o "Female")
        
    Returns:
        SpanishTTSResult con el resultado
    """
    tts = AdaptiveSpanishTTS()
    return tts.generate_with_best_available(text, output_path, voice_gender)

def get_available_spanish_voices() -> List[SpanishVoice]:
    """
    Función de conveniencia para obtener voces españolas disponibles
    
    Returns:
        Lista de voces españolas disponibles
    """
    voice_manager = SpanishVoiceManager()
    return voice_manager.get_available_voices()

def main():
    """Función para testing del módulo"""
    print("🎙️ TESTING TTS EN ESPAÑOL")
    print("=" * 40)
    
    # Mostrar voces disponibles
    voices = get_available_spanish_voices()
    print(f"Voces disponibles: {len(voices)}")
    
    for voice in voices[:3]:  # Mostrar solo las primeras 3
        print(f"  - {voice}")
    
    # Test de generación
    test_text = "Hola, este es un test de generación de voz en español."
    test_output = "test_audio.wav"
    
    print(f"\nGenerando audio de prueba...")
    result = generate_spanish_audio(test_text, test_output)
    
    if result.success:
        print(f"✅ Audio generado exitosamente")
        print(f"   Duración: {result.duration_seconds:.1f}s")
        print(f"   Tamaño: {result.file_size_mb:.1f}MB")
        print(f"   Voz: {result.voice_used}")
    else:
        print(f"❌ Error: {result.error_message}")

if __name__ == "__main__":
    main()