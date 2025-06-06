import os
import time
import subprocess
import gc
from pathlib import Path
from typing import Optional, Callable
from moviepy.video.io.VideoFileClip import VideoFileClip
from .interfaces import IAudioExtractor, IAudioSeparator, IAudioQualityAnalyzer
from .models import AudioExtractionResult, AudioSeparationResult, AudioProcessingProgress

class MoviePyAudioExtractor(IAudioExtractor):
    """Extractor de audio usando MoviePy + FFmpeg"""
    
    def extract_audio(self, video_path: str, output_path: str, format: str = "wav", sample_rate: int = 16000, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> AudioExtractionResult:
        """Extrae audio optimizado para Spleeter"""
        
        start_time = time.time()
        
        try:
            # Callback de progreso
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="extracting",
                    percentage=10,
                    current_step="Cargando video..."
                ))
            
            print("🎵 Cargando video...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                video.close()
                return AudioExtractionResult(
                    success=False,
                    error_message="El video no contiene audio"
                )
            
            # Progreso
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="extracting",
                    percentage=30,
                    current_step="Extrayendo audio temporal..."
                ))
            
            # Crear archivo temporal
            temp_path = output_path.replace(f".{format}", f"_temp.{format}")
            
            print("🔧 Extrayendo audio temporal...")
            video.audio.write_audiofile(temp_path, logger=None, verbose=False)
            
            duration = video.duration
            video.close()
            
            # Progreso
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="extracting",
                    percentage=60,
                    current_step="Optimizando para Spleeter..."
                ))
            
            print("🚀 Optimizando audio para Spleeter...")
            
            # Optimizar con FFmpeg para Spleeter
            comando = [
                "ffmpeg", "-y", "-i", temp_path,
                "-ac", "1",  # Mono (Spleeter funciona mejor)
                "-ar", str(sample_rate),  # Sample rate optimizado
                "-acodec", "pcm_s16le",  # WAV PCM 16-bit
                "-af", "volume=1.2",  # Ligero boost de volumen
                "-loglevel", "error",
                "-hide_banner",
                output_path
            ]
            
            subprocess.run(comando, check=True, capture_output=True)
            
            # Limpiar temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Progreso final
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="extracting",
                    percentage=100,
                    current_step="Audio extraído correctamente"
                ))
            
            # Calcular estadísticas
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            processing_time = time.time() - start_time
            
            print(f"✅ Audio extraído: {file_size_mb:.1f}MB en {processing_time:.1f}s")
            
            return AudioExtractionResult(
                success=True,
                audio_file_path=output_path,
                original_video_path=video_path,
                audio_format=format,
                sample_rate=sample_rate,
                duration_seconds=duration,
                file_size_mb=file_size_mb
            )
            
        except Exception as e:
            return AudioExtractionResult(
                success=False,
                error_message=f"Error extrayendo audio: {str(e)}"
            )

class SpleeterCLISeparator(IAudioSeparator):
    """Separador usando solo CLI de Spleeter - Sin dependencias problemáticas"""
    
    def __init__(self, model_name: str = "2stems-16kHz"):
        """Inicializar con modelo específico"""
        self.model_name = model_name
        self._verify_spleeter_cli()
    
    def _verify_spleeter_cli(self):
        """Verifica que Spleeter CLI funciona"""
        try:
            result = subprocess.run(
                ["spleeter", "--help"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✅ Spleeter CLI disponible")
            else:
                raise RuntimeError("Spleeter CLI no responde")
        except FileNotFoundError:
            raise RuntimeError("Spleeter no está instalado o no está en PATH")
        except Exception as e:
            raise RuntimeError(f"Error verificando Spleeter: {e}")
    
    def separate_audio(self, audio_path: str, output_directory: str, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> AudioSeparationResult:
        """Separación usando solo CLI - Evita problemas de dependencias"""
        
        start_time = time.time()
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(audio_path):
                return AudioSeparationResult(
                    success=False,
                    error_message=f"Archivo de audio no encontrado: {audio_path}"
                )
            
            # Crear directorio temporal para Spleeter
            temp_dir = os.path.join(output_directory, "temp_spleeter")
            os.makedirs(temp_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=10,
                    current_step="Preparando Spleeter..."
                ))
            
            print("🎵 Iniciando separación con Spleeter CLI...")
            print(f"📁 Audio: {os.path.basename(audio_path)}")
            print(f"🤖 Modelo: {self.model_name}")
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=20,
                    current_step="Ejecutando separación IA..."
                ))
            
            # Comando Spleeter CLI
            comando = [
                "spleeter", "separate",
                "-p", f"spleeter:{self.model_name}",
                "-o", temp_dir,
                audio_path
            ]
            
            print(f"🤖 Ejecutando: {' '.join(comando[:4])}...")
            
            # Ejecutar con timeout generoso para videos largos
            result = subprocess.run(
                comando,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hora timeout
            )
            
            if result.returncode != 0:
                return AudioSeparationResult(
                    success=False,
                    error_message=f"Spleeter falló: {result.stderr}"
                )
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=80,
                    current_step="Organizando archivos..."
                ))
            
            # Encontrar archivos generados por Spleeter
            base_name = Path(audio_path).stem
            spleeter_result_dir = os.path.join(temp_dir, base_name)
            
            if not os.path.exists(spleeter_result_dir):
                return AudioSeparationResult(
                    success=False,
                    error_message=f"Spleeter no generó el directorio esperado: {spleeter_result_dir}"
                )
            
            # Archivos generados por Spleeter
            vocals_temp = os.path.join(spleeter_result_dir, "vocals.wav")
            accompaniment_temp = os.path.join(spleeter_result_dir, "accompaniment.wav")
            
            # Verificar que los archivos existen
            if not os.path.exists(vocals_temp):
                return AudioSeparationResult(
                    success=False,
                    error_message="Spleeter no generó archivo vocals.wav"
                )
            
            if not os.path.exists(accompaniment_temp):
                return AudioSeparationResult(
                    success=False,
                    error_message="Spleeter no generó archivo accompaniment.wav"
                )
            
            # Crear nombres finales
            vocals_final = os.path.join(output_directory, f"{base_name}_vocals.wav")
            accompaniment_final = os.path.join(output_directory, f"{base_name}_accompaniment.wav")
            
            # Mover archivos a ubicación final
            import shutil
            shutil.move(vocals_temp, vocals_final)
            shutil.move(accompaniment_temp, accompaniment_final)
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=100,
                    current_step="Separación completada"
                ))
            
            # Calcular calidad básica
            quality_score = self._estimate_quality(vocals_final, accompaniment_final)
            
            print(f"✅ Separación Spleeter completada en {processing_time:.1f}s")
            print(f"   🎤 Vocals: {os.path.basename(vocals_final)}")
            print(f"   🎵 Música: {os.path.basename(accompaniment_final)}")
            print(f"   📊 Calidad estimada: {quality_score:.1%}")
            
            return AudioSeparationResult(
                success=True,
                vocals_path=vocals_final,
                accompaniment_path=accompaniment_final,
                original_audio_path=audio_path,
                separation_method=f"Spleeter-CLI-{self.model_name}",
                processing_time_seconds=processing_time,
                quality_score=quality_score
            )
            
        except subprocess.TimeoutExpired:
            return AudioSeparationResult(
                success=False,
                error_message="Timeout: Spleeter tardó más de 1 hora (posible problema con el archivo)"
            )
        except Exception as e:
            return AudioSeparationResult(
                success=False,
                error_message=f"Error en separación Spleeter: {str(e)}"
            )
    
    def _estimate_quality(self, vocals_path: str, accompaniment_path: str) -> float:
        """Estimación básica de calidad sin librerías adicionales"""
        try:
            # Verificar que ambos archivos existen y tienen contenido
            vocals_size = os.path.getsize(vocals_path)
            accompaniment_size = os.path.getsize(accompaniment_path)
            
            if vocals_size < 1000 or accompaniment_size < 1000:
                return 0.3  # Archivos muy pequeños = mala separación
            
            # Estimación basada en balance de tamaños
            total_size = vocals_size + accompaniment_size
            vocals_ratio = vocals_size / total_size
            
            # Score basado en balance (0.2-0.8 es bueno)
            if 0.2 <= vocals_ratio <= 0.8:
                return 0.85  # Buen balance
            elif 0.1 <= vocals_ratio <= 0.9:
                return 0.7   # Balance aceptable
            else:
                return 0.5   # Balance pobre
                
        except Exception:
            return 0.6  # Score neutral si no se puede analizar

class BasicFallbackSeparator(IAudioSeparator):
    """Separador de fallback usando solo FFmpeg - Sin dependencias extra"""
    
    def separate_audio(self, audio_path: str, output_directory: str, progress_callback: Optional[Callable[[AudioProcessingProgress], None]] = None) -> AudioSeparationResult:
        """Separación básica usando filtros de FFmpeg"""
        
        start_time = time.time()
        
        try:
            print("🔧 Usando separación FFmpeg básica (fallback)...")
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=20,
                    current_step="Separación básica con FFmpeg..."
                ))
            
            base_name = Path(audio_path).stem
            vocals_path = os.path.join(output_directory, f"{base_name}_vocals.wav")
            accompaniment_path = os.path.join(output_directory, f"{base_name}_accompaniment.wav")
            
            # Crear "vocals" removiendo frecuencias bajas (muy básico)
            vocals_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "highpass=f=200,lowpass=f=3000",  # Rango de voz humana
                "-ac", "1",
                vocals_path
            ]
            
            # Crear "accompaniment" removiendo frecuencias de voz (muy básico)  
            accompaniment_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "lowpass=f=200,highpass=f=3000",  # Fuera del rango de voz
                "-ac", "1", 
                accompaniment_path
            ]
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=50,
                    current_step="Procesando vocals..."
                ))
            
            # Ejecutar comandos
            subprocess.run(vocals_cmd, capture_output=True, check=True)
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=80,
                    current_step="Procesando música..."
                ))
            
            subprocess.run(accompaniment_cmd, capture_output=True, check=True)
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(AudioProcessingProgress(
                    stage="separating",
                    percentage=100,
                    current_step="Separación básica completada"
                ))
            
            print(f"✅ Separación FFmpeg completada en {processing_time:.1f}s")
            print("⚠️  Nota: Calidad básica - solo filtros de frecuencia")
            
            return AudioSeparationResult(
                success=True,
                vocals_path=vocals_path,
                accompaniment_path=accompaniment_path,
                original_audio_path=audio_path,
                separation_method="FFmpeg-Basic-Filters",
                processing_time_seconds=processing_time,
                quality_score=0.4  # Score bajo para método básico
            )
            
        except Exception as e:
            return AudioSeparationResult(
                success=False,
                error_message=f"Error en separación FFmpeg: {str(e)}"
            )