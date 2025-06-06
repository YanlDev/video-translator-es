"""
MÓDULO: COMPOSICIÓN DE VIDEO FINAL
==================================
Junta video original + audio en español + música de fondo = Video final
"""

import os
import time
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from .interfaces import (
    IVideoAnalyzer, IAudioMixer, IVideoComposer, IVideoEncoder,
    IProjectCompositionManager, IAdaptiveVideoComposer
)
from .models import (
    VideoAsset, AudioTrack, VideoCompositionRequest, VideoCompositionResult,
    CompositionProgress, CompositionTemplate, ProjectCompositionInfo
)

class FFmpegVideoAnalyzer(IVideoAnalyzer):
    """Analizador de video usando FFmpeg/FFprobe"""
    
    def analyze_video(self, video_path: str) -> VideoAsset:
        """Analiza archivo de video y retorna sus propiedades"""
        try:
            # Usar ffprobe para obtener información del video
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extraer información relevante
            format_info = info.get('format', {})
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video' and not video_stream:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and not audio_stream:
                    audio_stream = stream
            
            duration = float(format_info.get('duration', 0))
            resolution = ""
            if video_stream:
                width = video_stream.get('width', 0)
                height = video_stream.get('height', 0)
                resolution = f"{width}x{height}"
            
            return VideoAsset(
                asset_type="original_video",
                file_path=video_path,
                duration=duration,
                format=Path(video_path).suffix[1:],  # Sin el punto
                has_audio=audio_stream is not None,
                has_video=video_stream is not None,
                resolution=resolution
            )
            
        except Exception as e:
            print(f"Error analizando video: {e}")
            return VideoAsset(
                asset_type="original_video",
                file_path=video_path,
                duration=0.0,
                format="unknown",
                has_audio=False,
                has_video=False
            )
    
    def get_video_duration(self, video_path: str) -> float:
        """Obtiene duración del video"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def get_video_resolution(self, video_path: str) -> str:
        """Obtiene resolución del video"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def has_audio_track(self, video_path: str) -> bool:
        """Verifica si el video tiene audio"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return 'audio' in result.stdout
        except:
            return False

class FFmpegAudioMixer(IAudioMixer):
    """Mezclador de audio usando FFmpeg"""
    
    def mix_audio_tracks(self, audio_tracks: List[AudioTrack], output_path: str,
                        duration: float,
                        progress_callback: Optional[Callable[[CompositionProgress], None]] = None) -> bool:
        """Mezcla múltiples pistas de audio"""
        try:
            if not audio_tracks:
                return False
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="preparing",
                    percentage=10,
                    current_operation="Preparando mezcla de audio"
                ))
            
            # Construir comando FFmpeg para mezcla
            cmd = ['ffmpeg', '-y']
            
            # Agregar archivos de entrada
            for track in audio_tracks:
                cmd.extend(['-i', track.file_path])
            
            # Configurar filtros de audio
            filter_complex = self._build_audio_filter(audio_tracks, duration)
            
            if filter_complex:
                cmd.extend(['-filter_complex', filter_complex])
            
            # Configuración de salida
            cmd.extend([
                '-ac', '2',  # Estéreo
                '-ar', '44100',  # Sample rate
                '-c:a', 'aac',  # Codec AAC
                '-b:a', '192k',  # Bitrate
                '-t', str(duration),  # Duración
                output_path
            ])
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="composing",
                    percentage=50,
                    current_operation="Mezclando pistas de audio"
                ))
            
            # Ejecutar comando
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="finalizing",
                    percentage=100,
                    current_operation="Audio mezclado completado"
                ))
            
            return True
            
        except Exception as e:
            print(f"Error mezclando audio: {e}")
            return False
    
    def _build_audio_filter(self, audio_tracks: List[AudioTrack], duration: float) -> str:
        """Construye filtro complejo de FFmpeg para mezcla"""
        filter_parts = []
        input_refs = []
        
        for i, track in enumerate(audio_tracks):
            input_ref = f"[{i}:a]"
            
            # Aplicar volumen
            if track.volume != 1.0:
                filter_parts.append(f"{input_ref}volume={track.volume}[a{i}]")
                input_ref = f"[a{i}]"
            
            # Aplicar fade in/out si está configurado
            if track.fade_in > 0 or track.fade_out > 0:
                fade_filter = f"afade=t=in:st=0:d={track.fade_in}"
                if track.fade_out > 0:
                    fade_filter += f",afade=t=out:st={duration-track.fade_out}:d={track.fade_out}"
                filter_parts.append(f"{input_ref}{fade_filter}[af{i}]")
                input_ref = f"[af{i}]"
            
            input_refs.append(input_ref)
        
        # Mezclar todas las pistas
        if len(input_refs) > 1:
            mix_filter = "".join(input_refs) + f"amix=inputs={len(input_refs)}:duration=longest"
            filter_parts.append(mix_filter)
        
        return ";".join(filter_parts) if filter_parts else ""
    
    def apply_ducking(self, voice_track: str, background_track: str, output_path: str,
                     ducking_threshold: float = -20.0, ducking_ratio: float = 0.3) -> bool:
        """Aplica ducking (reduce música cuando hay voz)"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', background_track,  # [0] música
                '-i', voice_track,       # [1] voz
                '-filter_complex',
                f'[1:a]acompressor=threshold={ducking_threshold}dB:ratio={1/ducking_ratio}:attack=5:release=50[voice];'
                f'[0:a][voice]sidechaincompress=threshold={ducking_threshold}dB:ratio={ducking_ratio}:attack=5:release=50',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
            
        except Exception as e:
            print(f"Error aplicando ducking: {e}")
            return False
    
    def normalize_audio_levels(self, audio_path: str, target_level: float = -16.0) -> bool:
        """Normaliza niveles de audio"""
        try:
            # Crear archivo temporal
            temp_path = audio_path.replace('.wav', '_normalized.wav')
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-af', f'loudnorm=I={target_level}:TP=-1.5:LRA=11',
                '-c:a', 'aac',
                temp_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Reemplazar archivo original
            os.replace(temp_path, audio_path)
            return True
            
        except Exception as e:
            print(f"Error normalizando audio: {e}")
            return False

class FFmpegVideoComposer(IVideoComposer):
    """Compositor principal de video usando FFmpeg"""
    
    def __init__(self):
        self.video_analyzer = FFmpegVideoAnalyzer()
        self.audio_mixer = FFmpegAudioMixer()
    
    def compose_video(self, request: VideoCompositionRequest,
                     progress_callback: Optional[Callable[[CompositionProgress], None]] = None) -> VideoCompositionResult:
        """Compone video final combinando todos los elementos"""
        
        start_time = time.time()
        
        try:
            # Validar solicitud
            validation_errors = self.validate_composition_request(request)
            if validation_errors:
                return VideoCompositionResult(
                    success=False,
                    error_message=f"Errores de validación: {'; '.join(validation_errors)}"
                )
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="analyzing",
                    percentage=5,
                    current_operation="Analizando archivos de entrada"
                ))
            
            # Analizar video original
            original_video = self.video_analyzer.analyze_video(request.original_video_path)
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="preparing",
                    percentage=15,
                    current_operation="Preparando pistas de audio"
                ))
            
            # Preparar pistas de audio
            audio_tracks = self._prepare_audio_tracks(request, original_video.duration)
            
            # Crear audio mezclado
            mixed_audio_path = request.output_path.replace('.mp4', '_mixed_audio.wav')
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="composing",
                    percentage=30,
                    current_operation="Mezclando audio"
                ))
            
            audio_mixed = self.audio_mixer.mix_audio_tracks(
                audio_tracks, mixed_audio_path, original_video.duration, progress_callback
            )
            
            if not audio_mixed:
                return VideoCompositionResult(
                    success=False,
                    error_message="Error mezclando audio"
                )
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="encoding",
                    percentage=60,
                    current_operation="Componiendo video final"
                ))
            
            # Componer video final
            success = self._compose_final_video(
                request.original_video_path,
                mixed_audio_path,
                request.output_path,
                progress_callback
            )
            
            # Limpiar archivo temporal
            try:
                os.remove(mixed_audio_path)
            except:
                pass
            
            if not success:
                return VideoCompositionResult(
                    success=False,
                    error_message="Error componiendo video final"
                )
            
            # Analizar resultado final
            final_stats = self._analyze_final_video(request.output_path)
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(CompositionProgress(
                    stage="finalizing",
                    percentage=100,
                    current_operation="Video final completado"
                ))
            
            return VideoCompositionResult(
                success=True,
                final_video_path=request.output_path,
                duration_seconds=final_stats.get('duration', 0.0),
                file_size_mb=final_stats.get('file_size_mb', 0.0),
                video_resolution=final_stats.get('resolution', ''),
                audio_tracks_count=len(audio_tracks),
                video_codec=final_stats.get('video_codec', 'h264'),
                audio_codec=final_stats.get('audio_codec', 'aac'),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return VideoCompositionResult(
                success=False,
                processing_time=processing_time,
                error_message=f"Error inesperado: {str(e)}"
            )
    
    def _prepare_audio_tracks(self, request: VideoCompositionRequest, duration: float) -> List[AudioTrack]:
        """Prepara las pistas de audio para mezcla"""
        tracks = []
        
        # Pista principal: Audio en español
        spanish_track = AudioTrack(
            track_id="spanish_voice",
            name="Voz en Español",
            file_path=request.spanish_audio_path,
            track_type="spanish_voice",
            volume=request.spanish_voice_volume,
            is_primary=True
        )
        tracks.append(spanish_track)
        
        # Pista secundaria: Música de fondo (si existe)
        if request.background_music_path and os.path.exists(request.background_music_path):
            music_track = AudioTrack(
                track_id="background_music",
                name="Música de Fondo",
                file_path=request.background_music_path,
                track_type="background_music",
                volume=request.background_music_volume,
                fade_in=0.5,
                fade_out=1.0
            )
            tracks.append(music_track)
        
        return tracks
    
    def _compose_final_video(self, video_path: str, audio_path: str, output_path: str,
                           progress_callback: Optional[Callable] = None) -> bool:
        """Compone el video final combinando video original + audio mezclado"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,  # Video original
                '-i', audio_path,  # Audio mezclado
                '-c:v', 'copy',    # Copiar video sin recodificar (más rápido)
                '-c:a', 'aac',     # Codec de audio
                '-b:a', '192k',    # Bitrate de audio
                '-map', '0:v:0',   # Usar video del primer input
                '-map', '1:a:0',   # Usar audio del segundo input
                '-shortest',       # Duración del más corto
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
            
        except Exception as e:
            print(f"Error componiendo video final: {e}")
            return False
    
    def _analyze_final_video(self, video_path: str) -> Dict[str, Any]:
        """Analiza el video final generado"""
        try:
            video_asset = self.video_analyzer.analyze_video(video_path)
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            
            return {
                'duration': video_asset.duration,
                'file_size_mb': file_size_mb,
                'resolution': video_asset.resolution,
                'video_codec': 'h264',
                'audio_codec': 'aac'
            }
        except:
            return {}
    
    def validate_composition_request(self, request: VideoCompositionRequest) -> List[str]:
        """Valida solicitud de composición"""
        errors = []
        
        # Verificar archivos obligatorios
        if not os.path.exists(request.original_video_path):
            errors.append(f"Video original no encontrado: {request.original_video_path}")
        
        if not os.path.exists(request.spanish_audio_path):
            errors.append(f"Audio español no encontrado: {request.spanish_audio_path}")
        
        # Verificar música de fondo (opcional)
        if request.background_music_path and not os.path.exists(request.background_music_path):
            errors.append(f"Música de fondo no encontrada: {request.background_music_path}")
        
        # Verificar valores de volumen
        if not 0.0 <= request.spanish_voice_volume <= 1.0:
            errors.append("Volumen de voz española debe estar entre 0.0 y 1.0")
        
        if not 0.0 <= request.background_music_volume <= 1.0:
            errors.append("Volumen de música de fondo debe estar entre 0.0 y 1.0")
        
        return errors
    
    def estimate_processing_time(self, request: VideoCompositionRequest) -> float:
        """Estima tiempo de procesamiento"""
        try:
            duration = self.video_analyzer.get_video_duration(request.original_video_path)
            # Estimación: ~0.3x tiempo real para composición
            return duration * 0.3
        except:
            return 60.0  # Fallback: 1 minuto

class ProjectCompositionManager(IProjectCompositionManager):
    """Gestor para crear composiciones desde proyectos"""
    
    def create_project_composition(self, project_path: str) -> Optional[ProjectCompositionInfo]:
        """Crea información de composición desde estructura de proyecto"""
        try:
            assets = self.auto_detect_assets(project_path)
            
            if not assets['original_video'] or not assets['spanish_audio']:
                return None
            
            # Crear assets
            video_analyzer = FFmpegVideoAnalyzer()
            
            original_video = video_analyzer.analyze_video(assets['original_video'])
            original_video.asset_type = "original_video"
            
            spanish_audio = VideoAsset(
                asset_type="spanish_audio",
                file_path=assets['spanish_audio'],
                duration=self._get_audio_duration(assets['spanish_audio']),
                format="wav",
                has_audio=True,
                has_video=False
            )
            
            background_music = None
            if assets['background_music']:
                background_music = VideoAsset(
                    asset_type="background_music",
                    file_path=assets['background_music'],
                    duration=self._get_audio_duration(assets['background_music']),
                    format="wav",
                    has_audio=True,
                    has_video=False
                )
            
            project_name = Path(project_path).name
            
            return ProjectCompositionInfo(
                project_id=project_name,
                project_name=project_name,
                original_video=original_video,
                spanish_audio=spanish_audio,
                background_music=background_music,
                template_used=CompositionTemplate.create_standard_template()
            )
            
        except Exception as e:
            print(f"Error creando composición de proyecto: {e}")
            return None
    
    def auto_detect_assets(self, project_path: str) -> Dict[str, Optional[str]]:
        """Detecta automáticamente assets en proyecto"""
        assets = {
            'original_video': None,
            'spanish_audio': None,
            'background_music': None,
            'subtitles': None
        }
        
        project_path = Path(project_path)
        
        # Buscar video original en 1_original/
        original_dir = project_path / "1_original"
        if original_dir.exists():
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                video_files = list(original_dir.glob(f"*{ext}"))
                if video_files:
                    assets['original_video'] = str(video_files[0])
                    break
        
        # Buscar audio en español en 5_audio_es/
        audio_es_dir = project_path / "5_audio_es"
        if audio_es_dir.exists():
            for ext in ['.wav', '.mp3']:
                audio_files = list(audio_es_dir.glob(f"*{ext}"))
                if audio_files:
                    assets['spanish_audio'] = str(audio_files[0])
                    break
        
        # Buscar música de fondo en audio_separado/
        separated_dir = project_path / "audio_separado"
        if separated_dir.exists():
            accompaniment_files = list(separated_dir.glob("*accompaniment*.wav"))
            if accompaniment_files:
                assets['background_music'] = str(accompaniment_files[0])
        
        # Buscar subtítulos en 3_transcripcion/
        transcription_dir = project_path / "3_transcripcion"
        if transcription_dir.exists():
            srt_files = list(transcription_dir.glob("*.srt"))
            if srt_files:
                assets['subtitles'] = str(srt_files[0])
        
        return assets
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Obtiene duración de archivo de audio"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def validate_project_for_composition(self, project_path: str) -> List[str]:
        """Valida que proyecto esté listo para composición"""
        issues = []
        assets = self.auto_detect_assets(project_path)
        
        if not assets['original_video']:
            issues.append("No se encontró video original en 1_original/")
        
        if not assets['spanish_audio']:
            issues.append("No se encontró audio en español en 5_audio_es/")
        
        # Verificar que archivos existan y tengan contenido
        for asset_type, path in assets.items():
            if path and os.path.exists(path):
                if os.path.getsize(path) < 1000:  # Menos de 1KB
                    issues.append(f"Archivo {asset_type} muy pequeño: {path}")
        
        return issues

class AdaptiveVideoComposer(IAdaptiveVideoComposer):
    """Compositor adaptativo que maneja múltiples motores"""
    
    def __init__(self):
        self.composers = {
            'ffmpeg': FFmpegVideoComposer()
        }
        self.project_manager = ProjectCompositionManager()
    
    def compose_with_best_available(self, request: VideoCompositionRequest) -> VideoCompositionResult:
        """Compone usando el mejor compositor disponible"""
        
        available_composers = self.get_available_composers()
        
        if not available_composers:
            return VideoCompositionResult(
                success=False,
                error_message="No hay compositores disponibles"
            )
        
        # Intentar con FFmpeg (único por ahora)
        composer = self.composers['ffmpeg']
        return composer.compose_video(request)
    
    def get_available_composers(self) -> List[str]:
        """Retorna compositores disponibles"""
        available = []
        
        # Verificar FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            available.append('ffmpeg')
        except:
            pass
        
        return available
    
    def fallback_to_next_composer(self, failed_composer: str) -> Optional[IVideoComposer]:
        """Cambia al siguiente compositor (solo FFmpeg por ahora)"""
        return None

# Funciones de conveniencia para uso fácil
def compose_video_from_project(project_path: str, output_filename: Optional[str] = None) -> VideoCompositionResult:
    """
    Función de conveniencia para componer video desde proyecto
    
    Args:
        project_path: Ruta del proyecto
        output_filename: Nombre del archivo final (opcional)
        
    Returns:
        VideoCompositionResult con el resultado
    """
    composer = AdaptiveVideoComposer()
    project_manager = ProjectCompositionManager()
    
    # Detectar assets del proyecto
    project_info = project_manager.create_project_composition(project_path)
    if not project_info:
        return VideoCompositionResult(
            success=False,
            error_message="No se pudo crear composición desde el proyecto"
        )
    
    # Validar proyecto
    validation_errors = project_manager.validate_project_for_composition(project_path)
    if validation_errors:
        return VideoCompositionResult(
            success=False,
            error_message=f"Errores de validación: {'; '.join(validation_errors)}"
        )
    
    # Crear solicitud de composición
    if not output_filename:
        output_filename = f"{project_info.project_name}_final.mp4"
    
    output_path = os.path.join(project_path, "6_final", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    request = VideoCompositionRequest(
        project_name=project_info.project_name,
        original_video_path=project_info.original_video.file_path,
        spanish_audio_path=project_info.spanish_audio.file_path,
        background_music_path=project_info.background_music.file_path if project_info.background_music else None,
        output_path=output_path
    )
    
    return composer.compose_with_best_available(request)

def validate_project_ready_for_composition(project_path: str) -> List[str]:
    """
    Función de conveniencia para validar proyecto
    
    Args:
        project_path: Ruta del proyecto
        
    Returns:
        Lista de problemas encontrados (vacía si todo está bien)
    """
    project_manager = ProjectCompositionManager()
    return project_manager.validate_project_for_composition(project_path)
