#!/usr/bin/env python3
"""
Comprehensive Cost Estimation Analyzer for VideoQA System

This script analyzes the performance and costs of both offline and online processing
for the VideoQA system, including shot detection, vLLM feature extraction, vector indexing,
and real-time question answering.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scene_segmentation.pipeline import segment_video_into_scenes
from videoqa.shot_vector_indexer import ShotVectorIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class CostEstimator:
    """Comprehensive cost estimation for VideoQA system."""
    
    def __init__(self):
        # Hardware costs (monthly)
        self.hardware_costs = {
            "gpu_rtx_4090": {
                "hourly_rate": 2.5,  # $2.5/hour for RTX 4090 equivalent
                "monthly_rate": 1800,  # $1800/month for dedicated GPU
                "memory_gb": 24,
                "compute_capability": "high"
            },
            "gpu_rtx_3090": {
                "hourly_rate": 1.8,
                "monthly_rate": 1300,
                "memory_gb": 24,
                "compute_capability": "medium"
            },
            "cpu_only": {
                "hourly_rate": 0.5,
                "monthly_rate": 360,
                "memory_gb": 32,
                "compute_capability": "low"
            }
        }
        
        # Cloud hosting costs (monthly)
        self.cloud_costs = {
            "aws_g4dn_xlarge": {
                "hourly_rate": 0.526,
                "monthly_rate": 380,
                "gpu": "Tesla T4",
                "vCPU": 4,
                "memory_gb": 16
            },
            "aws_g5_xlarge": {
                "hourly_rate": 1.006,
                "monthly_rate": 725,
                "gpu": "A10G",
                "vCPU": 4,
                "memory_gb": 16
            },
            "google_gcp_n1_standard_4": {
                "hourly_rate": 0.19,
                "monthly_rate": 137,
                "gpu": "None",
                "vCPU": 4,
                "memory_gb": 15
            }
        }
        
        # Model performance characteristics
        self.model_performance = {
            "smolvlm": {
                "shots_per_hour": 120,  # 30 seconds per shot
                "memory_usage_gb": 8,
                "quality_score": 7.5,
                "cost_per_shot": 0.02,  # $0.02 per shot
                "batch_size": 4
            },
            "internvl_3_1b": {
                "shots_per_hour": 10,  # 6 minutes per shot
                "memory_usage_gb": 16,
                "quality_score": 9.0,
                "cost_per_shot": 0.15,  # $0.15 per shot
                "batch_size": 1
            },
            "llava_next": {
                "shots_per_hour": 30,  # 2 minutes per shot
                "memory_usage_gb": 12,
                "quality_score": 8.5,
                "cost_per_shot": 0.08,  # $0.08 per shot
                "batch_size": 2
            }
        }
        
        # Online processing costs
        self.online_costs = {
            "vector_search": {
                "latency_ms": 200,
                "cost_per_query": 0.001,
                "memory_usage_gb": 2
            },
            "vllm_inference": {
                "smolvlm": {
                    "latency_seconds": 3,
                    "cost_per_query": 0.01,
                    "memory_usage_gb": 8
                },
                "internvl_3_1b": {
                    "latency_seconds": 8,
                    "cost_per_query": 0.05,
                    "memory_usage_gb": 16
                },
                "llava_next": {
                    "latency_seconds": 5,
                    "cost_per_query": 0.03,
                    "memory_usage_gb": 12
                }
            }
        }
    
    def analyze_video_processing_performance(self, video_path: str, model_type: str = "smolvlm") -> Dict[str, Any]:
        """Analyze processing performance for a given video."""
        
        video_duration = self._get_video_duration(video_path)
        estimated_shots = self._estimate_shot_count(video_duration)
        
        # Get model performance data
        model_perf = self.model_performance[model_type]
        
        # Calculate processing times
        shot_detection_time = 2  # minutes, very fast
        vllm_processing_time = (estimated_shots / model_perf["shots_per_hour"]) * 60  # minutes
        vector_indexing_time = 2  # minutes
        
        total_processing_time = shot_detection_time + vllm_processing_time + vector_indexing_time
        
        return {
            "video_duration_minutes": video_duration / 60,
            "estimated_shots": estimated_shots,
            "shot_detection_time_minutes": shot_detection_time,
            "vllm_processing_time_minutes": vllm_processing_time,
            "vector_indexing_time_minutes": vector_indexing_time,
            "total_processing_time_minutes": total_processing_time,
            "total_processing_time_hours": total_processing_time / 60,
            "model_type": model_type,
            "shots_per_hour_rate": model_perf["shots_per_hour"],
            "memory_usage_gb": model_perf["memory_usage_gb"]
        }
    
    def calculate_offline_processing_costs(self, video_duration_hours: float, model_type: str = "smolvlm", 
                                         hardware_type: str = "gpu_rtx_4090") -> Dict[str, Any]:
        """Calculate offline processing costs for a given video duration."""
        
        # Get performance analysis
        performance = self.analyze_video_processing_performance("dummy.mp4", model_type)
        
        # Scale to target duration
        scale_factor = video_duration_hours / (performance["video_duration_minutes"] / 60)
        scaled_processing_time = performance["total_processing_time_hours"] * scale_factor
        scaled_shots = performance["estimated_shots"] * scale_factor
        
        # Get hardware costs
        hardware = self.hardware_costs[hardware_type]
        model_perf = self.model_performance[model_type]
        
        # Calculate costs
        hardware_cost = (scaled_processing_time * hardware["hourly_rate"])
        model_cost = scaled_shots * model_perf["cost_per_shot"]
        storage_cost = video_duration_hours * 0.1  # $0.1 per hour of video for storage
        
        total_cost = hardware_cost + model_cost + storage_cost
        
        return {
            "video_duration_hours": video_duration_hours,
            "estimated_shots": scaled_shots,
            "processing_time_hours": scaled_processing_time,
            "hardware_cost": hardware_cost,
            "model_cost": model_cost,
            "storage_cost": storage_cost,
            "total_cost": total_cost,
            "cost_per_hour_video": total_cost / video_duration_hours,
            "hardware_type": hardware_type,
            "model_type": model_type
        }
    
    def calculate_online_processing_costs(self, queries_per_month: int, model_type: str = "smolvlm",
                                        hosting_type: str = "aws_g4dn_xlarge") -> Dict[str, Any]:
        """Calculate online processing costs for Q&A service."""
        
        # Get hosting costs
        hosting = self.cloud_costs[hosting_type]
        vllm_costs = self.online_costs["vllm_inference"][model_type]
        vector_costs = self.online_costs["vector_search"]
        
        # Calculate costs
        hosting_cost = hosting["monthly_rate"]
        vllm_cost = queries_per_month * vllm_costs["cost_per_query"]
        vector_cost = queries_per_month * vector_costs["cost_per_query"]
        
        total_cost = hosting_cost + vllm_cost + vector_cost
        cost_per_query = total_cost / queries_per_month
        
        return {
            "queries_per_month": queries_per_month,
            "hosting_cost": hosting_cost,
            "vllm_cost": vllm_cost,
            "vector_cost": vector_cost,
            "total_cost": total_cost,
            "cost_per_query": cost_per_query,
            "hosting_type": hosting_type,
            "model_type": model_type,
            "avg_response_time_seconds": vllm_costs["latency_seconds"] + (vector_costs["latency_ms"] / 1000)
        }
    
    def generate_comprehensive_cost_report(self, video_hours_per_month: float = 10, 
                                         queries_per_month: int = 1000) -> Dict[str, Any]:
        """Generate a comprehensive cost report for the entire system."""
        
        report = {
            "summary": {
                "video_hours_per_month": video_hours_per_month,
                "queries_per_month": queries_per_month,
                "total_monthly_cost": 0
            },
            "offline_processing": {},
            "online_processing": {},
            "recommendations": []
        }
        
        # Calculate costs for different model types
        models = ["smolvlm", "llava_next", "internvl_3_1b"]
        
        for model in models:
            # Offline costs
            offline_costs = self.calculate_offline_processing_costs(
                video_hours_per_month, model, "gpu_rtx_4090"
            )
            
            # Online costs
            online_costs = self.calculate_online_processing_costs(
                queries_per_month, model, "aws_g4dn_xlarge"
            )
            
            total_monthly = offline_costs["total_cost"] + online_costs["total_cost"]
            
            report["offline_processing"][model] = offline_costs
            report["online_processing"][model] = online_costs
            
            if model == "smolvlm":  # Use SmolVLM as baseline
                report["summary"]["total_monthly_cost"] = total_monthly
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations."""
        
        recommendations = [
            "**Model Selection**: SmolVLM offers the best cost-performance ratio for most use cases",
            "**Batch Processing**: Process multiple videos simultaneously to reduce GPU idle time",
            "**Caching Strategy**: Implement aggressive caching for processed results",
            "**Auto-scaling**: Use cloud auto-scaling for online services to match demand",
            "**Storage Optimization**: Use compressed storage formats and lifecycle policies",
            "**GPU Optimization**: Use mixed precision training and efficient model loading",
            "**Query Caching**: Cache common queries to reduce vLLM inference costs",
            "**Load Balancing**: Distribute queries across multiple instances for better performance"
        ]
        
        return recommendations
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                "-of", "csv=p=0", video_path
            ], capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 300.0  # Default 5 minutes
    
    def _estimate_shot_count(self, video_duration_seconds: float) -> int:
        """Estimate number of shots based on video duration."""
        # Based on analysis: 77 shots in 264 seconds = ~0.29 shots/second
        shots_per_second = 0.29
        return int(video_duration_seconds * shots_per_second)

def print_cost_report(report: Dict[str, Any]):
    """Print a formatted cost report."""
    
    print("=" * 80)
    print("VIDEOQA SYSTEM COST ESTIMATION REPORT")
    print("=" * 80)
    
    summary = report["summary"]
    print(f"\n📊 SUMMARY")
    print(f"   Video Processing: {summary['video_hours_per_month']} hours/month")
    print(f"   Online Queries: {summary['queries_per_month']:,} queries/month")
    print(f"   Total Monthly Cost: ${summary['total_monthly_cost']:,.2f}")
    
    print(f"\n🔧 OFFLINE PROCESSING COSTS (per month)")
    print("-" * 50)
    
    for model, costs in report["offline_processing"].items():
        print(f"\n{model.upper()}:")
        print(f"   Processing Time: {costs['processing_time_hours']:.1f} hours")
        print(f"   Estimated Shots: {costs['estimated_shots']:,.0f}")
        print(f"   Hardware Cost: ${costs['hardware_cost']:.2f}")
        print(f"   Model Cost: ${costs['model_cost']:.2f}")
        print(f"   Storage Cost: ${costs['storage_cost']:.2f}")
        print(f"   Total Cost: ${costs['total_cost']:.2f}")
        print(f"   Cost per Hour: ${costs['cost_per_hour_video']:.2f}")
    
    print(f"\n🌐 ONLINE PROCESSING COSTS (per month)")
    print("-" * 50)
    
    for model, costs in report["online_processing"].items():
        print(f"\n{model.upper()}:")
        print(f"   Hosting Cost: ${costs['hosting_cost']:.2f}")
        print(f"   vLLM Cost: ${costs['vllm_cost']:.2f}")
        print(f"   Vector Search Cost: ${costs['vector_cost']:.2f}")
        print(f"   Total Cost: ${costs['total_cost']:.2f}")
        print(f"   Cost per Query: ${costs['cost_per_query']:.4f}")
        print(f"   Avg Response Time: {costs['avg_response_time_seconds']:.1f}s")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 50)
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "=" * 80)

def main():
    """Main function to run cost estimation analysis."""
    
    estimator = CostEstimator()
    
    # Generate comprehensive cost report
    print("Generating cost estimation report...")
    report = estimator.generate_comprehensive_cost_report(
        video_hours_per_month=10,  # 10 hours of video processing per month
        queries_per_month=1000     # 1000 queries per month
    )
    
    # Print the report
    print_cost_report(report)
    
    # Save detailed report to file
    with open("cost_estimation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: cost_estimation_report.json")
    
    # Additional analysis for 1-hour video specifically
    print(f"\n🎬 SPECIFIC ANALYSIS: 1-HOUR VIDEO PROCESSING")
    print("-" * 50)
    
    for model in ["smolvlm", "llava_next", "internvl_3_1b"]:
        costs = estimator.calculate_offline_processing_costs(1.0, model)
        print(f"\n{model.upper()}:")
        print(f"   Processing Time: {costs['processing_time_hours']:.1f} hours")
        print(f"   Total Cost: ${costs['total_cost']:.2f}")
        print(f"   Cost per Hour: ${costs['cost_per_hour_video']:.2f}")

if __name__ == "__main__":
    main() 