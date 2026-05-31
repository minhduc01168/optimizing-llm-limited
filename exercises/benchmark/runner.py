"""
Benchmark Runner Framework
===========================
Tự động thu thập metrics từ tất cả bài thực hành

Hardware: NVIDIA T4 (16GB VRAM)
"""

import json
import os
import time
import subprocess
from typing import Dict, List
from dataclasses import dataclass, asdict
from pathlib import Path
import glob


@dataclass
class LessonResult:
    """Kết quả từ mỗi lesson"""
    lesson_id: str
    lesson_name: str
    status: str  # "completed", "failed", "skipped"
    metrics_file: str
    execution_time_s: float
    key_metrics: Dict


class BenchmarkRunner:
    """Runner để thu thập metrics từ tất cả lessons"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results: List[LessonResult] = []
        self.output_dir = self.base_dir / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def discover_lessons(self) -> List[Path]:
        """Tìm tất cả lesson directories"""
        lessons = sorted(self.base_dir.glob("lesson-*"))
        return [l for l in lessons if l.is_dir()]
    
    def run_lesson(self, lesson_dir: Path) -> LessonResult:
        """Chạy benchmark cho một lesson"""
        lesson_id = lesson_dir.name
        starter_file = lesson_dir / "starter.py"
        
        if not starter_file.exists():
            return LessonResult(
                lesson_id=lesson_id,
                lesson_name=lesson_dir.name,
                status="skipped",
                metrics_file="",
                execution_time_s=0,
                key_metrics={}
            )
        
        print(f"\n{'='*60}")
        print(f"Running: {lesson_id}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Chạy starter.py
            result = subprocess.run(
                ["python", str(starter_file)],
                cwd=str(lesson_dir),
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Đọc metrics output
                metrics_file = lesson_dir / "results" / "metrics.json"
                key_metrics = {}
                
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        key_metrics = json.load(f)
                
                return LessonResult(
                    lesson_id=lesson_id,
                    lesson_name=lesson_dir.name,
                    status="completed",
                    metrics_file=str(metrics_file),
                    execution_time_s=execution_time,
                    key_metrics=key_metrics
                )
            else:
                print(f"Error: {result.stderr}")
                return LessonResult(
                    lesson_id=lesson_id,
                    lesson_name=lesson_dir.name,
                    status="failed",
                    metrics_file="",
                    execution_time_s=execution_time,
                    key_metrics={"error": result.stderr}
                )
                
        except subprocess.TimeoutExpired:
            return LessonResult(
                lesson_id=lesson_id,
                lesson_name=lesson_dir.name,
                status="failed",
                metrics_file="",
                execution_time_s=1800,
                key_metrics={"error": "Timeout after 30 minutes"}
            )
        except Exception as e:
            return LessonResult(
                lesson_id=lesson_id,
                lesson_name=lesson_dir.name,
                status="failed",
                metrics_file="",
                execution_time_s=time.time() - start_time,
                key_metrics={"error": str(e)}
            )
    
    def run_all(self):
        """Chạy benchmark cho tất cả lessons"""
        lessons = self.discover_lessons()
        print(f"Found {len(lessons)} lessons")
        
        for lesson in lessons:
            result = self.run_lesson(lesson)
            self.results.append(result)
        
        self.save_summary()
    
    def save_summary(self):
        """Lưu tổng hợp kết quả"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_lessons": len(self.results),
            "completed": sum(1 for r in self.results if r.status == "completed"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "skipped": sum(1 for r in self.results if r.status == "skipped"),
            "lessons": [asdict(r) for r in self.results]
        }
        
        output_file = self.output_dir / "benchmark_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {output_file}")
        self.print_summary()
    
    def print_summary(self):
        """In bảng tổng hợp"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"{'Lesson':<35} {'Status':<12} {'Time (s)':<12}")
        print("-"*70)
        
        for r in self.results:
            status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(r.status, "?")
            print(f"{r.lesson_name:<35} {status_icon} {r.status:<10} {r.execution_time_s:<12.1f}")
        
        print("-"*70)
        completed = sum(1 for r in self.results if r.status == "completed")
        print(f"Completed: {completed}/{len(self.results)}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all lessons")
    parser.add_argument("--lesson", type=str, help="Run specific lesson")
    parser.add_argument("--dir", type=str, default=".", help="Base directory")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.dir)
    
    if args.all:
        runner.run_all()
    elif args.lesson:
        lesson_dir = Path(args.dir) / args.lesson
        if lesson_dir.exists():
            result = runner.run_lesson(lesson_dir)
            runner.results.append(result)
            runner.save_summary()
        else:
            print(f"Lesson not found: {args.lesson}")
    else:
        print("Use --all to run all lessons or --lesson <name> for specific lesson")


if __name__ == "__main__":
    main()
