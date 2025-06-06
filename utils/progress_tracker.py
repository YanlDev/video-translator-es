"""
MÓDULO: SISTEMA DE SEGUIMIENTO DE PROGRESO
==========================================
Centraliza toda la lógica de progreso y barras de progreso
"""

import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ProgressInfo:
    """Información de progreso unificada"""
    current_step: str
    step_number: int
    total_steps: int
    percentage: float
    elapsed_time: float
    estimated_remaining: float
    stage: str = "processing"
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}
    
    def get_percentage_formatted(self) -> str:
        """Retorna porcentaje formateado"""
        return f"{self.percentage:.1f}%"
    
    def get_time_formatted(self) -> str:
        """Retorna tiempo formateado"""
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        elapsed = format_time(self.elapsed_time)
        remaining = format_time(self.estimated_remaining) if self.estimated_remaining > 0 else "?"
        
        return f"{elapsed} / ~{remaining}"

class ProgressDisplay(ABC):
    """Interface para mostrar progreso"""
    
    @abstractmethod
    def show_progress(self, progress: ProgressInfo) -> None:
        """Muestra el progreso actual"""
        pass
    
    @abstractmethod
    def show_completion(self, progress: ProgressInfo) -> None:
        """Muestra mensaje de finalización"""
        pass
    
    @abstractmethod
    def show_error(self, error_message: str) -> None:
        """Muestra mensaje de error"""
        pass

class ConsoleProgressDisplay(ProgressDisplay):
    """Muestra progreso en consola con barra visual"""
    
    def __init__(self, bar_width: int = 20, show_details: bool = True):
        self.bar_width = bar_width
        self.show_details = show_details
        self.last_line_length = 0
    
    def show_progress(self, progress: ProgressInfo) -> None:
        """Muestra barra de progreso en consola"""
        # Crear barra visual
        filled = "█" * int(progress.percentage * self.bar_width / 100)
        empty = "░" * (self.bar_width - len(filled))
        
        # Construir línea de progreso
        progress_line = f"\r🔄 [{filled}{empty}] {progress.get_percentage_formatted()}"
        
        if self.show_details:
            progress_line += f" - {progress.current_step}"
            
            if progress.stage:
                stage_emoji = self._get_stage_emoji(progress.stage)
                progress_line += f" {stage_emoji}"
            
            if progress.elapsed_time > 0:
                progress_line += f" ({progress.get_time_formatted()})"
        
        # Limpiar línea anterior si es necesaria
        if len(progress_line) < self.last_line_length:
            progress_line += " " * (self.last_line_length - len(progress_line))
        
        print(progress_line, end="", flush=True)
        self.last_line_length = len(progress_line)
    
    def show_completion(self, progress: ProgressInfo) -> None:
        """Muestra mensaje de finalización"""
        print(f"\n✅ Completado: {progress.current_step}")
        if progress.elapsed_time > 0:
            print(f"   ⏱️ Tiempo total: {time.strftime('%M:%S', time.gmtime(progress.elapsed_time))}")
    
    def show_error(self, error_message: str) -> None:
        """Muestra mensaje de error"""
        print(f"\n❌ Error: {error_message}")
    
    def _get_stage_emoji(self, stage: str) -> str:
        """Retorna emoji según el stage"""
        stage_emojis = {
            "downloading": "📥",
            "extracting": "🎵",
            "separating": "🎼",
            "transcribing": "🎤",
            "translating": "🌐",
            "generating": "🎙️",
            "composing": "🎬",
            "encoding": "⚙️",
            "finalizing": "🎉",
            "processing": "🔄",
            "analyzing": "🔍",
            "preparing": "🛠️",
            "saving": "💾"
        }
        return stage_emojis.get(stage, "🔄")

class SilentProgressDisplay(ProgressDisplay):
    """Display silencioso que no muestra nada"""
    
    def show_progress(self, progress: ProgressInfo) -> None:
        pass
    
    def show_completion(self, progress: ProgressInfo) -> None:
        pass
    
    def show_error(self, error_message: str) -> None:
        pass

class ProgressTracker:
    """Tracker de progreso para operaciones largas"""
    
    def __init__(self, total_steps: int, step_names: Optional[List[str]] = None, display: Optional[ProgressDisplay] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names = step_names or [f"Paso {i+1}" for i in range(total_steps)]
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.display = display or ConsoleProgressDisplay()
        self.step_times = []  # Para calcular mejores estimaciones
    
    def next_step(self, step_name: Optional[str] = None, stage: str = "processing") -> ProgressInfo:
        """Avanza al siguiente paso"""
        # Registrar tiempo del paso anterior
        if self.current_step > 0:
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
        
        self.current_step += 1
        self.step_start_time = time.time()
        
        # Actualizar nombre del paso si se proporciona
        if step_name and self.current_step <= len(self.step_names):
            self.step_names[self.current_step - 1] = step_name
        
        progress = self.get_progress(stage=stage)
        self.display.show_progress(progress)
        
        return progress
    
    def update_current_step(self, percentage: float, message: str = "", stage: str = "processing") -> ProgressInfo:
        """Actualiza el progreso del paso actual"""
        # Calcular porcentaje total considerando el paso actual
        steps_completed = self.current_step - 1
        step_progress = percentage / 100.0
        total_percentage = ((steps_completed + step_progress) / self.total_steps) * 100
        
        progress = ProgressInfo(
            current_step=message or self._get_current_step_name(),
            step_number=self.current_step,
            total_steps=self.total_steps,
            percentage=min(total_percentage, 100.0),
            elapsed_time=time.time() - self.start_time,
            estimated_remaining=self._estimate_remaining_time(total_percentage),
            stage=stage
        )
        
        self.display.show_progress(progress)
        return progress
    
    def get_progress(self, stage: str = "processing") -> ProgressInfo:
        """Obtiene información actual de progreso"""
        elapsed = time.time() - self.start_time
        percentage = (self.current_step / self.total_steps) * 100
        
        return ProgressInfo(
            current_step=self._get_current_step_name(),
            step_number=self.current_step,
            total_steps=self.total_steps,
            percentage=min(percentage, 100.0),
            elapsed_time=elapsed,
            estimated_remaining=self._estimate_remaining_time(percentage),
            stage=stage
        )
    
    def complete(self, final_message: str = "Proceso completado") -> ProgressInfo:
        """Marca el proceso como completado"""
        self.current_step = self.total_steps
        final_progress = ProgressInfo(
            current_step=final_message,
            step_number=self.total_steps,
            total_steps=self.total_steps,
            percentage=100.0,
            elapsed_time=time.time() - self.start_time,
            estimated_remaining=0.0,
            stage="completed"
        )
        
        self.display.show_completion(final_progress)
        return final_progress
    
    def error(self, error_message: str) -> None:
        """Marca el proceso como error"""
        self.display.show_error(error_message)
    
    def _get_current_step_name(self) -> str:
        """Obtiene nombre del paso actual"""
        if self.current_step == 0:
            return "Iniciando"
        elif self.current_step > self.total_steps:
            return "Completado"
        else:
            return self.step_names[self.current_step - 1]
    
    def _estimate_remaining_time(self, percentage: float) -> float:
        """Estima tiempo restante basado en progreso actual"""
        if percentage <= 0 or self.current_step == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        
        # Si tenemos histórico de tiempos de pasos, usar eso para mejor estimación
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            return avg_step_time * remaining_steps
        
        # Estimación simple basada en porcentaje
        if percentage <= 0 or self.current_step == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        
        # Si tenemos histórico de tiempos de pasos, usar eso para mejor estimación
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            return avg_step_time * remaining_steps
        
        # Estimación simple basada en porcentaje
        if percentage > 0:
            total_estimated = elapsed / (percentage / 100)
            return max(0, total_estimated - elapsed)
        
        return 0.0

class PipelineProgressTracker:
    """Tracker especializado para pipelines de video processing"""
    
    def __init__(self, pipeline_name: str = "Video Processing"):
        self.pipeline_name = pipeline_name
        self.phases = [
            ("Descarga", "downloading"),
            ("Extracción Audio", "extracting"),
            ("Separación Audio", "separating"),
            ("Transcripción", "transcribing"),
            ("Traducción", "translating"),
            ("Generación TTS", "generating"),
            ("Composición Final", "composing")
        ]
        
        self.current_phase = 0
        self.phase_progress = 0.0
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.display = ConsoleProgressDisplay()
        
        print(f"\n🚀 INICIANDO {self.pipeline_name.upper()}")
        print("=" * 60)
    
    def start_phase(self, phase_name: str = None, stage: str = None) -> ProgressInfo:
        """Inicia una nueva fase del pipeline"""
        if phase_name:
            # Buscar fase por nombre
            for i, (name, default_stage) in enumerate(self.phases):
                if phase_name in name or name in phase_name:
                    self.current_phase = i
                    break
        
        # Usar stage de la fase o el proporcionado
        if self.current_phase < len(self.phases):
            phase_name = phase_name or self.phases[self.current_phase][0]
            stage = stage or self.phases[self.current_phase][1]
        
        self.phase_start_time = time.time()
        self.phase_progress = 0.0
        
        print(f"\n🔄 Fase {self.current_phase + 1}/{len(self.phases)}: {phase_name}")
        
        progress = self._calculate_progress(stage)
        self.display.show_progress(progress)
        
        return progress
    
    def update_phase_progress(self, percentage: float, message: str = "", stage: str = None) -> ProgressInfo:
        """Actualiza progreso de la fase actual"""
        self.phase_progress = min(percentage, 100.0)
        
        if self.current_phase < len(self.phases):
            phase_name, default_stage = self.phases[self.current_phase]
            stage = stage or default_stage
            current_step = message or f"{phase_name} ({percentage:.1f}%)"
        else:
            current_step = message or f"Progreso: {percentage:.1f}%"
            stage = stage or "processing"
        
        progress = self._calculate_progress(stage, current_step)
        self.display.show_progress(progress)
        
        return progress
    
    def complete_phase(self, message: str = "") -> ProgressInfo:
        """Completa la fase actual"""
        self.phase_progress = 100.0
        
        if self.current_phase < len(self.phases):
            phase_name, stage = self.phases[self.current_phase]
            completion_message = message or f"{phase_name} completada"
        else:
            completion_message = message or "Fase completada"
            stage = "completed"
        
        progress = self._calculate_progress(stage, completion_message)
        
        # Mostrar tiempo de la fase
        phase_duration = time.time() - self.phase_start_time
        print(f"\n   ✅ Completada en {phase_duration:.1f}s")
        
        # Avanzar a siguiente fase
        self.current_phase += 1
        
        return progress
    
    def complete_pipeline(self, final_message: str = "") -> ProgressInfo:
        """Completa todo el pipeline"""
        total_time = time.time() - self.start_time
        
        final_message = final_message or f"{self.pipeline_name} completado"
        
        progress = ProgressInfo(
            current_step=final_message,
            step_number=len(self.phases),
            total_steps=len(self.phases),
            percentage=100.0,
            elapsed_time=total_time,
            estimated_remaining=0.0,
            stage="completed"
        )
        
        print(f"\n🎉 {final_message.upper()}")
        print(f"⏱️  Tiempo total: {time.strftime('%M:%S', time.gmtime(total_time))}")
        print("=" * 60)
        
        return progress
    
    def _calculate_progress(self, stage: str, current_step: str = "") -> ProgressInfo:
        """Calcula progreso total del pipeline"""
        # Progreso total = fases completadas + progreso de fase actual
        completed_phases = self.current_phase
        current_phase_contribution = self.phase_progress / 100.0
        
        total_progress = ((completed_phases + current_phase_contribution) / len(self.phases)) * 100
        
        elapsed = time.time() - self.start_time
        
        # Estimar tiempo restante
        if total_progress > 0:
            estimated_total = elapsed / (total_progress / 100)
            estimated_remaining = max(0, estimated_total - elapsed)
        else:
            estimated_remaining = 0.0
        
        # Nombre del paso actual
        if not current_step and self.current_phase < len(self.phases):
            current_step = self.phases[self.current_phase][0]
        
        return ProgressInfo(
            current_step=current_step,
            step_number=self.current_phase + 1,
            total_steps=len(self.phases),
            percentage=min(total_progress, 100.0),
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            stage=stage
        )

# Funciones de conveniencia

def create_simple_progress_tracker(total_steps: int, step_names: List[str] = None) -> ProgressTracker:
    """Crea un tracker de progreso simple"""
    return ProgressTracker(total_steps, step_names, ConsoleProgressDisplay())

def create_silent_progress_tracker(total_steps: int) -> ProgressTracker:
    """Crea un tracker de progreso silencioso"""
    return ProgressTracker(total_steps, display=SilentProgressDisplay())

def create_pipeline_tracker(pipeline_name: str = "Processing") -> PipelineProgressTracker:
    """Crea un tracker especializado para pipelines"""
    return PipelineProgressTracker(pipeline_name)

# Context manager para operaciones con progreso
class ProgressContext:
    """Context manager para operaciones con tracking de progreso automático"""
    
    def __init__(self, operation_name: str, estimated_duration: float = 0):
        self.operation_name = operation_name
        self.estimated_duration = estimated_duration
        self.tracker = ProgressTracker(1, [operation_name])
        self.start_time = None
    
    def __enter__(self):
        print(f"🔄 Iniciando: {self.operation_name}")
        self.start_time = time.time()
        self.tracker.next_step(self.operation_name, "processing")
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.tracker.complete(f"{self.operation_name} completado")
            print(f"✅ {self.operation_name} completado en {duration:.1f}s")
        else:
            self.tracker.error(f"Error en {self.operation_name}: {exc_val}")
            print(f"❌ Error en {self.operation_name} después de {duration:.1f}s")
        
        return False  # No suprimir excepciones