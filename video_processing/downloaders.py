import os
import yt_dlp
import re
from typing import Optional, Callable
from .interfaces import IVideoDownloader, IVideoInfoExtractor, IFileNameSanitizer, IProgressDisplay
from .models import VideoInfo, DownloadProgress, DownloadResult

class YouTubeInfoExtractor(IVideoInfoExtractor):
    """Extractor de información específico para YouTube"""
    
    def extract_info(self, url: str) -> Optional[VideoInfo]:
        """Extrae información del video de YouTube usando yt-dlp"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return VideoInfo(
                    titulo=info.get('title', 'Sin título'),
                    duracion=info.get('duration', 0),
                    canal=info.get('uploader', 'Desconocido'),
                    id=info.get('id', 'sin_id'),
                    url=url
                )
                
        except Exception as e:
            print(f"❌ Error obteniendo información: {e}")
            return None

class SimpleFileNameSanitizer(IFileNameSanitizer):
    """Limpiador simple de nombres de archivos"""
    
    def sanitize(self, filename: str, max_length: int = 30) -> str:
        """Limpia caracteres problemáticos del nombre"""
        # Eliminar caracteres problemáticos
        clean_name = re.sub(r'[<>:"/\\|?*\[\]{}()]', '_', filename)
        clean_name = re.sub(r'[^\w\s-]', '', clean_name)
        clean_name = ' '.join(clean_name.split())  # Eliminar espacios múltiples
        clean_name = clean_name.strip()
        clean_name = clean_name.replace(' ', '_')
        
        # Limitar longitud
        if len(clean_name) > max_length:
            clean_name = clean_name[:max_length]
        
        # Asegurar que no esté vacío
        if not clean_name:
            clean_name = "video"
            
        return clean_name
                
class ConsoleProgressDisplay(IProgressDisplay):
    """Muestra progreso en consola con barra visual"""
    
    def show_progress(self, progress: DownloadProgress) -> None:
        """Muestra barra de progreso en consola"""
        percentage = progress.percentage
        filled = "█" * int(percentage // 5)
        empty = "░" * (20 - int(percentage // 5))
        
        print(f"\r🔄 [{filled}{empty}] {percentage:5.1f}% - {progress.message}", 
              end="", flush=True)
        
        if progress.status == 'finished':
            print()

class YouTubeDownloader(IVideoDownloader):
    """Descargador específico para YouTube usando yt-dlp"""
    
    def __init__(self, file_sanitizer: IFileNameSanitizer, progress_display: IProgressDisplay):
        """Constructor con inyección de dependencias"""
        self.file_sanitizer = file_sanitizer
        self.progress_display = progress_display
    
    def download(self, url: str, output_path: str, 
                progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> DownloadResult:
        """Descarga video de YouTube"""
        
        # Extraer info del video
        info_extractor = YouTubeInfoExtractor()
        video_info = info_extractor.extract_info(url)
        
        if not video_info:
            return DownloadResult(
                success=False,
                error_message="No se pudo obtener información del video"
            )
        
        # Crear nombre seguro
        safe_filename = self.file_sanitizer.sanitize(video_info.titulo)
        filename_with_id = f"{safe_filename}_{video_info.id}"
        
        # Configurar yt-dlp
        ydl_opts = {
            'format': 'best[height<=1080][ext=mp4]/best[height<=1080]/best',
            'outtmpl': os.path.join(output_path, f'{filename_with_id}.%(ext)s'),
            'noplaylist': True,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._create_progress_hook(progress_callback)],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            downloaded_file = self._find_downloaded_file(output_path, filename_with_id)
            
            return DownloadResult(
                success=True,
                file_path=downloaded_file,
                video_info=video_info
            )
            
        except Exception as e:
            return DownloadResult(
                success=False,
                error_message=f"Error en descarga: {str(e)}",
                video_info=video_info
            )
    
    def _create_progress_hook(self, callback):
        """Crea el hook de progreso para yt-dlp"""
        def hook(d):
            if d['status'] == 'downloading':
                if 'total_bytes' in d:
                    percentage = (d['downloaded_bytes'] / d['total_bytes']) * 100
                    total = d['total_bytes']
                elif 'total_bytes_estimate' in d:
                    percentage = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                    total = d['total_bytes_estimate']
                else:
                    return
                
                progress = DownloadProgress(
                    percentage=percentage,
                    downloaded_bytes=d['downloaded_bytes'],
                    total_bytes=total,
                    status='downloading'
                )
                
                self.progress_display.show_progress(progress)
                if callback:
                    callback(progress)
            
            elif d['status'] == 'finished':
                progress = DownloadProgress(
                    percentage=100.0,
                    downloaded_bytes=d.get('downloaded_bytes', 0),
                    total_bytes=d.get('total_bytes', 0),
                    status='finished',
                    message="Completado"
                )
                
                self.progress_display.show_progress(progress)
                if callback:
                    callback(progress)
        
        return hook
    
    def _find_downloaded_file(self, output_path: str, filename_base: str) -> Optional[str]:
        """Encuentra el archivo descargado"""
        extensions = ['.mp4', '.webm', '.mkv', '.avi']
    
        for ext in extensions:
            file_path = os.path.join(output_path, f"{filename_base}{ext}")
            if os.path.exists(file_path):
                return file_path
        return None
