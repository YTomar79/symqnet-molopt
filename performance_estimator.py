"""
Performance Estimator for Universal SymQNet
Calculates expected performance degradation for non-10-qubit systems
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance degradation levels"""
    OPTIMAL = "optimal"
    EXCELLENT = "excellent"  
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    SEVERE = "severe"

@dataclass
class PerformanceReport:
    """Performance assessment report"""
    n_qubits: int
    performance_factor: float  # 0.0 to 1.0
    level: PerformanceLevel
    warning_message: Optional[str]
    recommendations: List[str]
    uncertainty_scaling: float
    computational_overhead: float

class PerformanceEstimator:
    """
    Estimates performance degradation for Universal SymQNet
    Based on distance from optimal 10-qubit training size
    """
    
    def __init__(self, optimal_qubits: int = 10):
        self.optimal_qubits = optimal_qubits
        self.min_viable_qubits = 2
        self.performance_cache = {}
        
        # Empirical performance models (can be tuned based on experiments)
        self.degradation_models = self._initialize_degradation_models()
        
        logger.info(f"Performance estimator initialized - optimal at {optimal_qubits} qubits")
    
    def _initialize_degradation_models(self) -> Dict[str, Dict]:
        """Initialize empirical performance degradation models"""
        
        return {
            'small_systems': {
                # For systems smaller than optimal (padding effects)
                'base_degradation': 0.95,  # Base degradation per qubit difference
                'padding_penalty': 0.02,   # Additional penalty for information dilution
                'min_performance': 0.7     # Minimum performance for small systems
            },
            'large_systems': {
                # For systems larger than optimal (compression effects)
                'base_degradation': 0.90,  # Base degradation per qubit difference  
                'compression_penalty': 0.03, # Additional penalty for information loss
                'scaling_exponent': 1.2,   # Non-linear degradation for large systems
                'min_performance': 0.4     # Minimum performance for large systems
            },
            'uncertainty_scaling': {
                # How uncertainty estimates are affected
                'small_systems_factor': 1.2,  # Uncertainty increases for small systems
                'large_systems_factor': 1.5,  # Uncertainty increases more for large systems
                'base_uncertainty': 0.1       # Base uncertainty at optimal size
            },
            'computational_overhead': {
                # Additional computational cost factors
                'normalization_overhead': 0.1,    # 10% overhead for normalization
                'compression_overhead_base': 0.05, # Base overhead for compression
                'compression_overhead_scale': 0.02 # Additional overhead per extra qubit
            }
        }
    
    def estimate_performance(self, n_qubits: int, 
                           include_recommendations: bool = True) -> PerformanceReport:
        """
        Estimate performance for given qubit count
        
        Args:
            n_qubits: Number of qubits in target system
            include_recommendations: Whether to include optimization recommendations
            
        Returns:
            Comprehensive performance report
        """
        
        # Check cache first
        cache_key = f"{n_qubits}_{include_recommendations}"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # Validate input
        if n_qubits < self.min_viable_qubits:
            raise ValueError(f"Minimum {self.min_viable_qubits} qubits required, got {n_qubits}")
        
        # Calculate performance factor
        performance_factor = self._calculate_performance_factor(n_qubits)
        
        # Determine performance level
        level = self._get_performance_level(performance_factor)
        
        # Generate warning message
        warning_message = self._generate_warning_message(n_qubits, performance_factor, level)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(n_qubits, performance_factor, level)
        
        # Calculate uncertainty scaling
        uncertainty_scaling = self._calculate_uncertainty_scaling(n_qubits)
        
        # Calculate computational overhead
        computational_overhead = self._calculate_computational_overhead(n_qubits)
        
        # Create report
        report = PerformanceReport(
            n_qubits=n_qubits,
            performance_factor=performance_factor,
            level=level,
            warning_message=warning_message,
            recommendations=recommendations,
            uncertainty_scaling=uncertainty_scaling,
            computational_overhead=computational_overhead
        )
        
        # Cache result
        self.performance_cache[cache_key] = report
        
        return report
    
    def _calculate_performance_factor(self, n_qubits: int) -> float:
        """Calculate performance factor (0.0 to 1.0) for given qubit count"""
        
        if n_qubits == self.optimal_qubits:
            return 1.0
        
        distance = abs(n_qubits - self.optimal_qubits)
        
        if n_qubits < self.optimal_qubits:
            # Small systems: information dilution from padding
            models = self.degradation_models['small_systems']
            
            base_perf = models['base_degradation'] ** distance
            padding_penalty = models['padding_penalty'] * distance
            
            performance = base_perf - padding_penalty
            performance = max(performance, models['min_performance'])
            
        else:
            # Large systems: information loss from compression
            models = self.degradation_models['large_systems']
            
            base_perf = models['base_degradation'] ** (distance ** models['scaling_exponent'])
            compression_penalty = models['compression_penalty'] * distance
            
            performance = base_perf - compression_penalty
            performance = max(performance, models['min_performance'])
        
        return min(performance, 1.0)
    
    def _get_performance_level(self, performance_factor: float) -> PerformanceLevel:
        """Categorize performance factor into levels"""
        
        if performance_factor >= 0.98:
            return PerformanceLevel.OPTIMAL
        elif performance_factor >= 0.90:
            return PerformanceLevel.EXCELLENT
        elif performance_factor >= 0.80:
            return PerformanceLevel.GOOD
        elif performance_factor >= 0.65:
            return PerformanceLevel.MODERATE
        elif performance_factor >= 0.45:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.SEVERE
    
    def _generate_warning_message(self, n_qubits: int, performance_factor: float, 
                                level: PerformanceLevel) -> Optional[str]:
        """Generate appropriate warning message"""
        
        if level == PerformanceLevel.OPTIMAL:
            return None
        
        if level == PerformanceLevel.EXCELLENT:
            return (f"Minimal performance impact expected for {n_qubits}-qubit system "
                   f"({performance_factor:.1%} of optimal)")
        
        elif level == PerformanceLevel.GOOD:
            return (f"Small performance degradation for {n_qubits}-qubit system "
                   f"({performance_factor:.1%} of optimal). Results remain reliable.")
        
        elif level == PerformanceLevel.MODERATE:
            return (f"Moderate performance degradation for {n_qubits}-qubit system "
                   f"({performance_factor:.1%} of optimal). Consider validation against known results.")
        
        elif level == PerformanceLevel.POOR:
            return (f"Significant performance degradation for {n_qubits}-qubit system "
                   f"({performance_factor:.1%} of optimal). Results may be unreliable - "
                   f"use with caution and validate thoroughly.")
        
        else:  # SEVERE
            return (f"Severe performance degradation for {n_qubits}-qubit system "
                   f"({performance_factor:.1%} of optimal). Results likely unreliable - "
                   f"consider alternative methods or system size reduction.")
    
    def _generate_recommendations(self, n_qubits: int, performance_factor: float,
                                level: PerformanceLevel) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if level == PerformanceLevel.OPTIMAL:
            recommendations.append("System at optimal size - maximum accuracy expected")
            recommendations.append("Use standard parameters for best results")
        
        elif n_qubits < self.optimal_qubits:
            # Small systems
            recommendations.append(f"Consider expanding to {self.optimal_qubits}-qubit active space for optimal accuracy")
            recommendations.append("Increase number of rollouts to compensate for padding effects")
            recommendations.append("Use higher shot counts to improve statistics")
            
            if level in [PerformanceLevel.POOR, PerformanceLevel.SEVERE]:
                recommendations.append("Consider using traditional methods for this system size")
        
        else:
            # Large systems
            recommendations.append(f"Consider reducing to {self.optimal_qubits}-qubit active space if possible")
            recommendations.append("Use more rollouts to account for compression artifacts")
            recommendations.append("Validate results against known benchmarks")
            
            if level == PerformanceLevel.MODERATE:
                recommendations.append("Consider freezing core orbitals to reduce system size")
            elif level in [PerformanceLevel.POOR, PerformanceLevel.SEVERE]:
                recommendations.append("Strongly recommend system size reduction or alternative methods")
                recommendations.append("Current system size may exceed reliable prediction range")
        
        # General recommendations based on performance level
        if level in [PerformanceLevel.MODERATE, PerformanceLevel.POOR, PerformanceLevel.SEVERE]:
            recommendations.append("Use results for qualitative analysis rather than quantitative predictions")
            recommendations.append("Cross-validate with traditional quantum chemistry methods")
        
        return recommendations
    
    def _calculate_uncertainty_scaling(self, n_qubits: int) -> float:
        """Calculate how uncertainty estimates should be scaled"""
        
        if n_qubits == self.optimal_qubits:
            return 1.0
        
        models = self.degradation_models['uncertainty_scaling']
        distance = abs(n_qubits - self.optimal_qubits)
        
        if n_qubits < self.optimal_qubits:
            # Small systems have increased uncertainty due to padding
            scaling = 1.0 + (models['small_systems_factor'] - 1.0) * (distance / self.optimal_qubits)
        else:
            # Large systems have increased uncertainty due to compression
            scaling = 1.0 + (models['large_systems_factor'] - 1.0) * (distance / self.optimal_qubits)
        
        return scaling
    
    def _calculate_computational_overhead(self, n_qubits: int) -> float:
        """Calculate additional computational overhead factor"""
        
        if n_qubits == self.optimal_qubits:
            return 1.0  # No overhead at optimal size
        
        models = self.degradation_models['computational_overhead']
        base_overhead = models['normalization_overhead']
        
        if n_qubits > self.optimal_qubits:
            # Additional overhead for compression
            extra_qubits = n_qubits - self.optimal_qubits
            compression_overhead = (models['compression_overhead_base'] + 
                                  models['compression_overhead_scale'] * extra_qubits)
            total_overhead = base_overhead + compression_overhead
        else:
            # Only normalization overhead for small systems
            total_overhead = base_overhead
        
        return 1.0 + total_overhead
    
    def get_performance_curve(self, qubit_range: Tuple[int, int] = (2, 25)) -> Dict[int, float]:
        """Generate performance curve for range of qubit counts"""
        
        min_qubits, max_qubits = qubit_range
        curve = {}
        
        for n_qubits in range(min_qubits, max_qubits + 1):
            performance = self._calculate_performance_factor(n_qubits)
            curve[n_qubits] = performance
        
        return curve
    
    def get_recommended_parameters(self, n_qubits: int) -> Dict[str, Union[int, float]]:
        """Get recommended CLI parameters for given system size"""
        
        report = self.estimate_performance(n_qubits)
        
        # Base parameters
        base_shots = 1024
        base_rollouts = 5
        base_max_steps = 50
        
        # Adjust based on performance level
        if report.level == PerformanceLevel.OPTIMAL:
            recommended = {
                'shots': base_shots,
                'n_rollouts': base_rollouts,
                'max_steps': base_max_steps,
                'confidence': 0.95
            }
        
        elif report.level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
            # Slight increase for non-optimal systems
            recommended = {
                'shots': int(base_shots * 1.2),
                'n_rollouts': base_rollouts + 2,
                'max_steps': base_max_steps,
                'confidence': 0.95
            }
        
        elif report.level == PerformanceLevel.MODERATE:
            # Significant increase to compensate for degradation
            recommended = {
                'shots': int(base_shots * 1.5),
                'n_rollouts': base_rollouts + 5,
                'max_steps': int(base_max_steps * 1.2),
                'confidence': 0.90  # Lower confidence due to uncertainty
            }
        
        else:  # POOR or SEVERE
            # Maximum parameters to get any reasonable results
            recommended = {
                'shots': int(base_shots * 2.0),
                'n_rollouts': base_rollouts + 10,
                'max_steps': int(base_max_steps * 1.5),
                'confidence': 0.80  # Much lower confidence
            }
        
        return recommended
    
    def warn_if_needed(self, n_qubits: int, warn_threshold: float = 0.9) -> None:
        """Issue warning if performance is below threshold"""
        
        report = self.estimate_performance(n_qubits, include_recommendations=False)
        
        if report.performance_factor < warn_threshold and report.warning_message:
            warnings.warn(report.warning_message, UserWarning, stacklevel=2)
    
    def print_performance_summary(self, n_qubits: int) -> None:
        """Print comprehensive performance summary"""
        
        report = self.estimate_performance(n_qubits)
        
        print(f"\n🔍 PERFORMANCE ANALYSIS: {n_qubits}-QUBIT SYSTEM")
        print("=" * 60)
        print(f"📊 Performance Factor: {report.performance_factor:.1%}")
        print(f"🎯 Performance Level: {report.level.value.upper()}")
        print(f"📈 Uncertainty Scaling: {report.uncertainty_scaling:.1f}x")
        print(f"⚡ Computational Overhead: {report.computational_overhead:.1f}x")
        
        if report.warning_message:
            print(f"\n⚠️  WARNING: {report.warning_message}")
        
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Show recommended parameters
        params = self.get_recommended_parameters(n_qubits)
        print(f"\n🎛️  RECOMMENDED PARAMETERS:")
        for param, value in params.items():
            print(f"   --{param.replace('_', '-')}: {value}")
    
    def benchmark_performance_range(self, qubit_range: Tuple[int, int] = (4, 20)) -> None:
        """Print performance benchmark table"""
        
        min_qubits, max_qubits = qubit_range
        
        print(f"\n📊 PERFORMANCE BENCHMARK TABLE")
        print("=" * 70)
        print(f"{'Qubits':<8} {'Performance':<12} {'Level':<12} {'Uncertainty':<12} {'Overhead':<10}")
        print("-" * 70)
        
        for n_qubits in range(min_qubits, max_qubits + 1):
            report = self.estimate_performance(n_qubits, include_recommendations=False)
            
            level_short = report.level.value[:8]  # Truncate for display
            
            print(f"{n_qubits:<8} {report.performance_factor:<12.1%} "
                  f"{level_short:<12} {report.uncertainty_scaling:<12.1f}x "
                  f"{report.computational_overhead:<10.1f}x")
        
        print("-" * 70)
        print(f"Optimal performance at {self.optimal_qubits} qubits")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_performance_check(n_qubits: int, optimal_qubits: int = 10) -> float:
    """Quick performance factor calculation without full report"""
    estimator = PerformanceEstimator(optimal_qubits)
    return estimator._calculate_performance_factor(n_qubits)

def get_performance_warning(n_qubits: int, optimal_qubits: int = 10) -> Optional[str]:
    """Get warning message for given qubit count"""
    estimator = PerformanceEstimator(optimal_qubits)
    report = estimator.estimate_performance(n_qubits, include_recommendations=False)
    return report.warning_message

def print_performance_curve(qubit_range: Tuple[int, int] = (4, 20), optimal_qubits: int = 10) -> None:
    """Print ASCII performance curve"""
    estimator = PerformanceEstimator(optimal_qubits)
    curve = estimator.get_performance_curve(qubit_range)
    
    print(f"\n📈 PERFORMANCE CURVE (Optimal: {optimal_qubits} qubits)")
    print("=" * 60)
    
    for qubits, performance in curve.items():
        bar_length = int(performance * 40)  # 40-char bar
        bar = "█" * bar_length + "░" * (40 - bar_length)
        marker = " ★" if qubits == optimal_qubits else ""
        print(f"{qubits:2d} qubits: {bar} {performance:.1%}{marker}")


